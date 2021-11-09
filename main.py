import copy
import time
import time
import numpy as np
import argparse

from torch.utils import data

from utils.utils import *
from utils import load_config
from utils.validate import *
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas

def train(model, config, logger, record):
    # initialize user_ids
    user_ids = np.arange(0, config.users)
    num_participators = int(config.part_rate*config.users) 

    # load the dataset
    dataset = assign_user_data(config, logger)
    test_images = torch.from_numpy(dataset["test_data"]["images"]).to(config.device)
    test_labels = torch.from_numpy(dataset["test_data"]["labels"]).to(config.device)

    # tau candidates 
    taus = np.array(config.taus)

    # before optimization, report the result first
    validate_and_log(model, dataset, config, record, logger)

    # start communication training rounds
    for comm_round in range(config.rounds):
        logger.info("Comm Round {:d}".format(comm_round))

        np.random.shuffle(user_ids)
        participator_ids = user_ids[:num_participators]

        # schedule some taus to pick up
        acc = []
        losses = []
        params_list = []

        global_kernel = None
        global_xs = None
        global_ys = None
        local_packages = []
        local_kernels = []
        for user_id in participator_ids:
            # print("user {:d} updating".format(user_id))
            
            user_resource = assign_user_resource(config, user_id, 
                                dataset["train_data"], dataset["user_with_data"])
            local_updater = LocalUpdater(config, user_resource)
            
            local_updater.local_step(model)
            local_package = local_updater.uplink_transmit()
            local_packages.append(local_package)

            if global_xs is None:
                global_xs = local_updater.xs
                global_ys = local_updater.ys
            else:
                global_xs = torch.vstack((global_xs, local_updater.xs))
                global_ys = torch.vstack((global_ys, local_updater.ys))            

            # del local_updater
            torch.cuda.empty_cache()

        start_time = time.time()

        global_jac = combine_local_jacobians(local_packages)

        del local_packages

        print("compute kernel matrix")
        global_kernel = empirical_kernel(global_jac)

        print("kernel computation time {:3f}".format(time.time() - start_time))

        predictor = gradient_descent_ce(global_kernel.cpu(), global_ys.cpu(), config.lr)
        
        with torch.no_grad():
            fx_0 = model(global_xs)

        t = torch.arange(config.taus[-1]+1)
        fx_train = predictor(t, fx_0.cpu())
        # fx_train = fx_train.to(fx_0)

        init_state_dict = copy.deepcopy(model.state_dict())

        losses = np.zeros_like(taus, dtype=float)
        acc = np.zeros_like(taus, dtype=float)

        print("loss \tacc")

        for i, tau in enumerate(config.taus):
            weight_aggregator = WeightMod(init_state_dict)
            global_omegas = get_omegas(t[:tau+1], config.lr, global_jac, 
                    global_ys.cpu(), fx_train[:tau+1], config.loss, 
                    model.state_dict())
            # global_omegas = get_omegas(t[:tau+1], config.lr, global_jac, 
            #         global_ys, fx_train[:tau+1], config.loss, 
            #         model.state_dict())        

            weight_aggregator.add(global_omegas)
            aggregated_weight = weight_aggregator.state_dict()
            model.load_state_dict(aggregated_weight)

            output = model(global_xs)

            loss = loss_with_output(output, global_ys, config.loss)
            # loss_fx = loss_with_output(fx_train[tau].to(global_ys), global_ys, config.loss)
            losses[i] = loss

            output = model(test_images)

            test_acc = accuracy_with_output(output, test_labels)
            acc[i] = test_acc

            print("{:.3f}\t{:.3f}".format(loss, test_acc))

            params_list.append(copy.deepcopy(aggregated_weight))

        idx = np.argmin(losses)
        
        params = params_list[idx]

        current_tau = taus[idx]
        current_acc = acc[idx]
        current_loss = losses[idx]

        logger.info("current tau {:d}".format(current_tau))
        logger.info("acc {:4f}".format(current_acc))
        logger.info("loss {:.4f}".format(current_loss))
        model.load_state_dict(params)

        # del params_list

        record["loss"].append(current_loss)
        record["testing_accuracy"].append(current_acc)
        record["taus"].append(current_tau)

        logger.info("-"*80)
        torch.cuda.empty_cache()

def main(config_file):
    config = load_config(config_file)
    logger = init_logger(config)
    
    model = init_model(config, logger)

    record = init_record(config, model)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    start = time.time()
    train(model, config, logger, record)
    end = time.time()
    save_record(config, record)
    logger.info("{:.3f} mins has elapsed.".format((end-start)/60))

if __name__ == "__main__":
    main("config.yaml")