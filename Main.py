import numpy as np
import random

from sklearn import metrics
import time
from sklearn import preprocessing
import torch

import DGCN
import LSG
import scipy.io as sio
import scipy.io as scio
from thop import profile, clever_format

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][1]

for (FLAG, curr_train_ratio,Scale) in [(1, 200, 100)]:

    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL=[]
    Test_Time_ALL=[]

    Seed_List=[0]

    if FLAG == 1:
        data_mat = sio.loadmat('HyperImage_data\\GF5\\GF5_462_617.mat')
        data = data_mat['data']
        gt_mat = sio.loadmat('HyperImage_data\\GF5\\gt_GF5_462_617.mat')
        gt = gt_mat['label']

        val_ratio = 200
        class_count = 8
        learning_rate = 5e-4
        max_epoch = 600
        dataset_name = "GF5_"
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('HyperImage_data\\Chris\\510_511.mat')
        data = data_mat['data']
        gt_mat = sio.loadmat('HyperImage_data\\Chris\\gt510_511.mat')
        gt = gt_mat['label']

        val_ratio = 200
        class_count = 6
        learning_rate = 5e-4
        max_epoch = 600
        dataset_name = "Chris_"
        pass

    if FLAG == 3:
        data_mat = sio.loadmat('HyperImage_data\\GT9\\data.mat')
        data = data_mat['data']
        gt_mat = sio.loadmat('HyperImage_data\\GT9\\gt.mat')
        gt = gt_mat['label']

        val_ratio = 0.05
        class_count = 9
        learning_rate = 5e-4
        max_epoch = 600
        dataset_name = "GT9_"
        pass

    if FLAG == 4:
        data_mat = sio.loadmat('HyperImage_data\\KSC\\KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('HyperImage_data\\KSC\\KSC_gt.mat')
        gt = gt_mat['KSC_gt']

        val_ratio = 0.1
        class_count = 13
        learning_rate = 5e-4
        max_epoch = 600
        dataset_name = "KSC_"
        pass

    superpixel_scale=Scale
    train_samples_per_class = curr_train_ratio  # 训练样本个数
    val_samples = class_count
    train_ratio = curr_train_ratio
    m, n, d = data.shape

    orig_data=data
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    
    def GT_To_One_Hot(gt, class_count):
        GT_One_Hot = []
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count,dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int( gt[i, j]) - 1] = 1
                GT_One_Hot.append(temp)
        GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
        return GT_One_Hot
   
    for curr_seed in Seed_List:
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':

            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount*train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx, dtype=object)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)


            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            

            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx
            

            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index
            

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
        
        if samples_type == 'same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class = train_samples_per_class
                rand_list = [i for i in range(samplesCount)]
                if real_train_samples_per_class > samplesCount:
                    real_train_samples_per_class = samplesCount
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            val_data_count = int(val_ratio)
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            
            test_data_index = test_data_index - val_data_index
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass

        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_GT = np.reshape(test_samples_gt, [m, n])

        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt=np.reshape(train_samples_gt,[height,width])
        test_samples_gt=np.reshape(test_samples_gt,[height,width])
        val_samples_gt=np.reshape(val_samples_gt,[height,width])

        train_samples_gt_onehot=GT_To_One_Hot(train_samples_gt,class_count)
        test_samples_gt_onehot=GT_To_One_Hot(test_samples_gt,class_count)
        val_samples_gt_onehot=GT_To_One_Hot(val_samples_gt,class_count)

        train_samples_gt_onehot=np.reshape(train_samples_gt_onehot,[-1,class_count]).astype(int)  # 转化为calss_count列 (21025,16)
        test_samples_gt_onehot=np.reshape(test_samples_gt_onehot,[-1,class_count]).astype(int)
        val_samples_gt_onehot=np.reshape(val_samples_gt_onehot,[-1,class_count]).astype(int)

        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m* n, class_count])

        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m* n, class_count])

        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m* n, class_count])

        # learnable superpixel generation
        lsg = LSG.LSG(data, train_samples_gt_onehot)
        tic0 = time.time()
        Q, S, A, Seg = lsg.simple_superpixel(scale=superpixel_scale)
        toc0 = time.time()
        LSG_Time = toc0 - tic0

        print("LSG costs time: {}".format(LSG_Time))
        Q = Q.to(device)
        A = torch.from_numpy(A).to(device)

        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)
        
        
        net_input=np.array( data,np.float32)
        net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)

        net = DGCN.DGCN(height, width, bands, class_count, Q, A)

        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)

        def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot
            we = -torch.mul(real_labels,torch.log(predict))
            we = torch.mul(we, reallabel_mask)
            pool_cross_entropy = torch.sum(we)
            return pool_cross_entropy
        

        zeros = torch.zeros([m * n]).to(device).float()
        def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
            if False==require_AA_KPP:
                with torch.no_grad():
                    available_label_idx=(train_samples_gt!=0).float()
                    available_label_count=available_label_idx.sum()
                    correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                    OA= correct_prediction.cpu()/available_label_count

                    return OA
            else:
                with torch.no_grad():
                    available_label_idx=(train_samples_gt!=0).float()
                    available_label_count=available_label_idx.sum()
                    correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                    OA= correct_prediction.cpu()/available_label_count
                    OA=OA.cpu().numpy()

                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt = train_samples_gt.cpu().numpy()

                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    # 输出
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)

                    # 保存数据信息
                    f = open('results\\' + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class:' + str(test_AC_list) + "\n"
                                  # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                  # + '\ntest time:' + str(time_test_end - time_test_start) \
                    f.write(str_results)
                    f.close()
                    return OA

        # 训练
        print("start train")
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        best_loss = 99999
        net.train()
        tic1 = time.clock()
        flops, params = profile(net.cuda(), inputs=(net_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print("fops, params:", flops, params)
        for i in range(max_epoch+1):
            optimizer.zero_grad()
            output = net(net_input)
            loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
            loss.backward(retain_graph=True)
            optimizer.step()  # Does the update
            if i%10==0:
                with torch.no_grad():
                    net.eval()
                    output= net(net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

                    if valloss < best_loss :
                        best_loss = valloss
                        torch.save(net.state_dict(),"model\\best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.clock()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time=toc1 - tic1 + LSG_Time
        Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model\\best_model.pt"))
            net.eval()
            tic2 = time.clock()
            output = net(net_input)
            toc2 = time.clock()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
            classification_map = np.reshape(classification_map, [-1])
            classification_map[list(background_idx)] = 0
            classification_map = np.reshape(classification_map, [m, n])
            classification_map1 = classification_map.numpy()
            scio.savemat(dataset_name+str(testOA)+'_pre.mat', {'label': classification_map1})
            testing_time = toc2 - tic2 + LSG_Time
            Test_Time_ALL.append(testing_time)
            
        torch.cuda.empty_cache()
        del net
        
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)

    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))  # np.std为标准差
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
