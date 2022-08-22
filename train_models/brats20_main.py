#!/usr/bin/env python

import wandb

wandb.init(project=' brats20-main-model-train', entity="huang-xin")

import os
import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as T

from dataset_loader.brats20_datasetLoader import BraTS20Dataset
from models.brats20_model_r18_mil import mulMod3D_MIL
from models.brats20_model_r18_mil_pe import mul3D_MIL_pe

from scipy import stats

import statistics
from  sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error 

import pandas as pd 


def medianSE(loss):  # median of SE
    result = statistics.median(loss)
    return result


def stdSE(loss): # standard deviation of SE
    result = statistics.stdev(loss)
    return result

def spearman_R(targets, prediction): #spearman's rank
    results, _ = stats.spearmanr(targets, prediction, axis=None) # = rho, pval
    results = round(results, 3) # around 0.001
    return results

def prepare_dataset(brats2020_dataset, batch_size):
    train_size = int(0.70 * len(brats2020_dataset))
    test_size = int(0.20 * len(brats2020_dataset))
    val_size = len(brats2020_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(brats2020_dataset, [train_size, test_size, val_size])
    # prepare brats2020 dataset
    trainloader = DataLoader(dataset=train_dataset,  batch_size= batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(dataset=test_dataset,  batch_size= batch_size, shuffle=False, num_workers=1)
    valloader = DataLoader(dataset=val_dataset,  batch_size= batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader, valloader


def errors_functions(targets, prediction, loss):
    mse = medianSE(loss)
    stdse = stdSE(loss)
    spr = spearman_R(targets, prediction)
    #r2 = r2_score(targets, prediction)
    return mse, stdse, spr

def normal_modality(modality):
    if modality == "flair":
        nor_modality = T.Normalize(46.04, 142.00)
    elif modality == "t1":
        nor_modality = T.Normalize(82.55, 297.99)
    elif modality == "t1ce":
        nor_modality = T.Normalize(103.04, 434.33)
    elif modality == "t2":
        nor_modality = T.Normalize(93.91, 348.19)

    return nor_modality

#test the dataset
def run_test(save_file_name, save_pos_file_name, valloader, name='Test'):
    # using model for testing dataset
    model.eval()
    #early stopping
    last_loss = 1000
    patience = 5
    trigger_times = 0
    with torch.no_grad():
        print('====================start test================')
        val_targets_list = []
        val_outputs_list = []

        #save attention weights in a data frame
        df = pd.DataFrame()
        #save predict results in a dataframe
        df_os = pd.DataFrame()
        sample_id_list = []
        pre_list = []

        for data_val in valloader:
            inputs_val, target_val, sample_id= data_val
            inputs_val = inputs_val.float()
            target_val = target_val.float()
            val_targets_list.append(target_val.numpy())

            target_val = target_val.resize_(1, 1).to(device)

            inputs_val = inputs_val.to(device)
            pred_val, attention_weights = model(inputs_val)
            val_outputs_list.append(pred_val.cpu().detach().numpy().ravel())
            
            #save weights of sample in a dictionay
            atten_weights = torch.squeeze(attention_weights, 1)              
            data_sample_weights = list(atten_weights.cpu().detach().numpy().ravel())
            df_id = sample_id[0]
            df_data = {df_id: data_sample_weights}
            df[df_id] = data_sample_weights 

            #add predicte value to the list, only for one patient
            output_val = torch.squeeze(pred_val, 1) 
            op_val_list = output_val.cpu().detach().numpy().ravel()
            pre_list.append(op_val_list[0])

            #save sample id
            sample_id_list.append(df_id)

            # loss_v = loss_function_MSE(pred_val, target_val)
            loss_v = loss_MSE(pred_val, target_val)

            #early stopping 
            if loss_v > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    break 
            else:
                trigger_times = 0 
            last_loss = loss_v 

            loss_vl.append(round(loss_v.item(), 3))

        #save weights
        df.to_csv(save_file_name, index=False)
        #save predicted os results
        df_os['Brats20ID'] = sample_id_list
        df_os['Predicted_survival_time'] = pre_list
        df_os.to_csv(save_pos_file_name, index=False)

        # print("For validation datasets, Epoch: {}, \n mean of  Loss: {:.3f}, \n "
        #         "MAE loss: {}".format(epoch + 1, round(np.mean(loss_vl), 3), loss_vl))

        mse_val = round(np.mean(loss_vl), 3)
        wandb.log({"epoch": epoch + 1,
                    name + "_loss ": mse_val})
        # print("validation loss", mse_val)

        mae_val = mean_absolute_error(val_targets_list, val_outputs_list)
        mse_val = mean_squared_error(val_targets_list, val_outputs_list)
        wandb.log({"epoch": epoch+1,
                name + "_MAE": round(mae_val, 3),
                name + "_MSE": round(mse_val, 3),})
        
        loss_medianMSE, loss_stdMSE, loss_spr = errors_functions(val_targets_list, val_outputs_list, loss_vl)
        r2_train = r2_score(val_targets_list, val_outputs_list)
        wandb.log({"epoch": epoch+1,
                    name + "stdMSE " : loss_stdMSE,
                    name + "medianMSE " : loss_medianMSE
                    })
        wandb.log({"epoch": epoch + 1,
                name + " R2 ": round(r2_train, 3)})
        wandb.log({"epoch": epoch + 1,
                name + "spearman_R": loss_spr})


def metrics(targets_list, output_list, loss_list, name="Train"):
            mse_train = mean_squared_error(targets_list, outputs_list)
            mae_train = mean_absolute_error(targets_list, outputs_list)
            loss_medianMSE, loss_stdMSE, loss_spr = errors_functions(targets_list, outputs_list, loss_list)
            r2_train = r2_score(targets_list, outputs_list)

            wandb.log({"epoch": epoch+1,
                    "MSE " + name : round(mse_train, 3),
                    "stdMSE " + name : loss_stdMSE,
                    "medianMSE " + name : loss_medianMSE
                    })
            wandb.log({"epoch": epoch+1,
                    "MAE " + name : round(mae_train, 3)})
            wandb.log({"epoch": epoch + 1,
                    " R2 " + name : round(r2_train, 3)})
            wandb.log({"epoch": epoch + 1,
                    "spearman_R" + name : loss_spr})


if  __name__ == "__main__":

    #set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 130
    learning_rate = 0.00001

    modality = "t1ce"  # !! rename #
    batch_size = 1
    save_model_path = '/dhc/home/huang.xin/brats20_slurms/train_slurms/run_main_r18_mil_pe/t1ce_r18_mil_pe_main_set3.pth' #!! rename #
    # model = mulMod3D_MIL(1, 'resnet18', heatmaps=True).to(device)
    model = mul3D_MIL_pe(1, 'resnet18' , heatmaps=True, device=device).to(device)

    #save files for train dataset
    save_file_name = "main_" + modality + "_train_r18_mil_pe_w.csv"
    save_pos_file_name = "main_" + modality + "_train_r18_mil_pe_os.csv"

    #save files for validation dataset 
    save_w_file_val = "main_" + modality + "_val_r18_mil_pe_w.csv"
    save_pos_file_val = "main_" + modality + "_val_r18_mil_pe_os.csv"

    #save files for test dataset 
    save_w_file_test = "main_" + modality + "_test_r18_mil_pe_w.csv"
    save_pos_file_test = "main_" + modality + "_test_r18_mil_pe_os.csv"

    brats20_train_path = "/dhc/home/huang.xin/BraTS2020/train_20/"

    # all training dataset
    survival_info_file = "survival_info.csv"
    train_set_file = 'train_set_3.csv'
    val_set_file = 'val_set_3.csv'
    test_set_file = 'test_set_3.csv'

    #118 samples of GTR in training dataset
    #survival_info_file = 'survival_info_GTR_samples.csv'

    transforms_train = torch.nn.Sequential(
        T.RandomHorizontalFlip(p=0.2),
        T.RandomRotation(20),
        T.RandomVerticalFlip(p=0.2),
        T.CenterCrop(160),
        normal_modality(modality)

    )

    transforms_val = torch.nn.Sequential(
        normal_modality(modality)
    )

    train_dataset = BraTS20Dataset(brats20_train_path, train_set_file, multimod_list=[modality], transform=transforms_train, get_sample_id=True)
    val_dataset = BraTS20Dataset(brats20_train_path, val_set_file, multimod_list=[modality], transform=transforms_val, get_sample_id=True)
    test_dataset = BraTS20Dataset(brats20_train_path, test_set_file, multimod_list=[modality], transform=transforms_val, get_sample_id=True)


    trainloader = DataLoader(dataset=train_dataset,  batch_size= batch_size, shuffle=True, num_workers=1)
    valloader = DataLoader(dataset=val_dataset,  batch_size= batch_size, shuffle=False, num_workers=1)
    testloader = DataLoader(dataset=test_dataset,  batch_size= batch_size, shuffle=False, num_workers=1)

    #loading dataset without data augmentation
    # transforms = torch.nn.Sequential(normal_modality(modality))
    # brats2020_dataset = BraTS20Dataset(brats20_train_path, survival_info_file,
    #                                multimod_list=[modality], transform=transforms, get_sample_id=True)
    # trainloader, testloader, valloader  = prepare_dataset(brats2020_dataset, batch_size)

    ## Initialize the model
    wandb.watch(model)

    # define the loss functions
    loss_MSE = nn.MSELoss()  # the mean squared error(MSE)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        wandb.log({'epoch': epoch + 1 })

        #print epoch
        print(f'Starting epoch =================== {epoch + 1}')

        # Set current loss value
        current_loss = 0.0
        current_loss_mae = 0.0

        #outputs list
        targets_list = []
        outputs_list = []

        loss_list = [] #training set
        loss_ls = []  #testing set
        loss_vl = []  #validation set

        ##save attention weights in a data frame
        df = pd.DataFrame()
        #save predict results in a dataframe
        df_os = pd.DataFrame()
        sample_id_list = []
        pre_list = []

        # Iterate over the DataLoader for training data
        model.train()
        for i, data in enumerate(trainloader, 0):

            # get and prepare inputs
            inputs, targets, sample_id = data
            inputs = inputs.float()
            targets = targets.float()

            #save targets in a list
            targets_list.append(targets.numpy())

            targets = targets.resize_(1, 1).to(device)

            targets_np = targets.cpu().detach().numpy()

            inputs = inputs.to(device)

            # zero the gradients
            optimizer.zero_grad()
        
            # perform forward pass
            outputs, attention_weights = model(inputs)

            # save outputs in the list
            outputs_list.append(outputs.cpu().detach().numpy().ravel())

            #save weights of sample in a dictionay
            atten_weights = torch.squeeze(attention_weights, 1)              
            data_sample_weights = list(atten_weights.cpu().detach().numpy().ravel())
            df_id = sample_id[0]
            df_data = {df_id: data_sample_weights}
            df[df_id] = data_sample_weights 

            #add predicte value to the list, only for one patient
            output_train = torch.squeeze(outputs, 1) 
            op_train_list = output_train.cpu().detach().numpy().ravel()
            pre_list.append(op_train_list[0])

            #save sample id
            sample_id_list.append(df_id)

            loss  = loss_MSE(outputs, targets)

            loss.backward()

            optimizer.step()

            # print statistics
            current_loss += loss.item()

            loss_list.append(loss.item())

        mse_loss_train = round(np.mean(loss_list), 3)
        wandb.log({"epoch": epoch + 1,
                    "train_loss ": mse_loss_train})

        #save weights
        df.to_csv(save_file_name, index=False)
        #save predicted os results
        df_os['Brats20ID'] = sample_id_list
        df_os['Predicted_survival_time'] = pre_list
        df_os.to_csv(save_pos_file_name, index=False)

        #metrics for train dataset 
        metrics(targets_list, outputs_list, loss_list, name='train')

        #for validation dataset
        run_test(save_w_file_val, save_pos_file_val, valloader, name='validation')
        

    #for test dataset
    run_test(save_w_file_test, save_pos_file_test, testloader, name='test')


    # process is complete
    print('Training process has finished, thanks!')

    #Save the model on the CPU, load on GPU
    PATH = '/dhc/home/huang.xin/my_project/models_brats2020'

    #save model 
    torch.save(
        {'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'Epoch': num_epochs,
        'learning rate': learning_rate,
        'modalities': modality
        }, save_model_path)






