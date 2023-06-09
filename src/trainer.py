import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import cross_transformer 
from models import conv1d
from models import conv2d
from models import conv3d
from models import cross_domain
import utils
from augment import do_augment
from utils import recorder
from evaluation import HSIEvaluation
import random

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import itertools


class SKlearnTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.evalator = HSIEvaluation(param=params)


        self.model = None
        self.real_init()

    def real_init(self):
        pass
        

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY)
        print(self.model, "trian done.") 


    def final_eval(self, testX, testY):
        predictY = self.model.predict(testX)
        temp_res = self.evalator.eval(testY, predictY)
        print(temp_res['oa'], temp_res['aa'], temp_res['kappa'])
        return temp_res

    def test(self, testX):
        return self.model.predict(testX)

            
class SVMTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super(SVMTrainer, self).__init__(params)

    def real_init(self):
        kernel = self.net_params.get('kernel', 'rbf')
        gamma = self.net_params.get('gamma', 'scale')
        c = self.net_params.get('c', 1)
        self.model = svm.SVC(C=c, kernel=kernel, gamma=gamma)

class RandomForestTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n_estimators = self.net_params.get('n_estimators', 200)
        self.model = RandomForestClassifier(n_estimators = n_estimators, max_features="auto", criterion="entropy")

class KNNTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n = self.net_params.get('n', 10)
        self.model = KNeighborsClassifier(n_neighbors=n)



class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.evalator = HSIEvaluation(param=params)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.augment=params.get('augment',None)
        self.clip = 50
        self.real_init()
        self.temp_unlabel_loader = None

    def real_init(self):
        pass

    def next_unalbel_data(self): 
        index, (data, target) = next(self.temp_unlabel_loader)
        print(index)
        target = torch.ones_like(target) * -1
        return data.to(self.device), target.to(self.device)

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
        
    def train(self, train_loader, test_loader=None):
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.augment:
                    data=do_augment(self.augment,data)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
            
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        y_pred_test, y_test = self.test(test_loader)
        temp_res = self.evalator.eval(y_test, y_pred_test)
        return temp_res


    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.get_logits(self.net(inputs))
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test

class BaseContraTrainer(BaseTrainer):
    def __init__(self,params):
        super(BaseContraTrainer,self).__init__(params)
        self.aug1=params.get("aug1",None)
        self.aug2=params.get("aug2",None)
    
    def train(self, train_loader, test_loader=None):
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (label_data, real_target,unlabel_data,minus_1) in enumerate(train_loader):
                label_data, real_target,unlabel_data,minus_1 = label_data.to(self.device), real_target.to(self.device),unlabel_data.to(self.device),minus_1.to(self.device)
                '''
                label_data、unlabelled_data大小：
                    batch_size channel height width
                real_target、-1向量大小：
                    batch_size
                是否需要做一个shuffle？其实做不做都行吧，先不做，之后向陈总请教
                '''
                data=torch.cat([label_data,unlabel_data],dim=0)
                target=torch.cat([real_target,minus_1],dim=0)
                data1=do_augment(self.aug1,data).to(self.device)
                data2=do_augment(self.aug2,data).to(self.device)
                outputs1 = self.net(data1)
                outputs2 = self.net(data2)
                outputs=outputs1+(outputs2[1],)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
            
        print('Finished Training')
        return True



class CrossTransformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(CrossTransformerTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = cross_transformer.HSINet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class ContraCrossTransformerTrainer(BaseContraTrainer):
    def __init__(self, params):
        super(ContraCrossTransformerTrainer,self).__init__(params)

    def real_init(self):
        # net
        self.net = cross_transformer.HSINet(self.params).to(self.device)
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def infoNCE_diag(self, A_vecs, B_vecs,mask, temperature=10):
        '''
        targets: [batch]  dtype is int
        mask: [batch] 元素只有01，分别表示labelled数据和unlabelled数据。前者应该被摒弃
        '''
        # print(A_vecs, B_vecs)
        # mask=mask.reshape(A_vecs.size(0),-1)
        batch_size=A_vecs.size(0)
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        # A内部之间也有负样本，使用相乘提出
        C_vecs=torch.cat([A_vecs,B_vecs],dim=0)
        # matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        matrix_logits = torch.matmul(C_vecs, torch.transpose(C_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        # mask_mat=mask*mask.transpose(0,1)
        # matrix_softmax = torch.softmax(matrix_logits, dim=1)*mask_mat # softmax by dim=1
        matrix_exp = torch.exp(matrix_logits) # softmax by dim=1
        sum_but_diag=torch.sum(matrix_exp,dim=1)-torch.diag(matrix_exp)
        # sum_but_diag=torch.Tensor.expand(sum_but_diag,C_vecs.size())
        matrix_softmax=matrix_exp.div(sum_but_diag)
        tempb = matrix_softmax.detach().cpu().numpy()
        # diag=np.diag(tempb)
        # print(np.diag(tempb))
        # print("softmax,", tempb.max(), tempb.min())
        matrix_log = -1 * torch.log(matrix_softmax)
        # here just use dig part
        # 对于matrix_log，进行四分块，只取右上块的对角线，是正例
        loss_nce=0.
        for i in range(batch_size):
            loss_nce+=matrix_log[i,batch_size+i]
        # loss_nce = torch.mean(torch.diag(matrix_log))
        loss_nce/=batch_size
        return loss_nce

    def infoNCE(self, A_vecs, B_vecs, targets, temperature=15):
        '''
        targets: [batch]  dtype is int
        '''
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        # tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        matrix_softmax = torch.softmax(matrix_logits, dim=1) # softmax by dim=1
        tempb = matrix_softmax.detach().cpu().numpy()
        # print("softmax,", tempb)
        # print("label,", targets)
        matrix_log = -1 * torch.log(matrix_softmax)

        l = targets.shape[0]
        tb = torch.repeat_interleave(targets.reshape([-1,1]), l, dim=1)
        tc = torch.repeat_interleave(targets.reshape([1,-1]), l, dim=0)
        mask_matrix = tb.eq(tc).int()
        # here just use dig part
        loss_nce = torch.sum(matrix_log * mask_matrix) / torch.sum(mask_matrix)
        return loss_nce



    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch(取值class_num)]
            target: [batch(取值class_num（包含-1）)]
            target里面包含-1的，是unlabel_data 通过mask完成两部分分开的计算
        '''
        logits, A_vecs, B_vecs = outputs
        # print("A_vecs Size:")
        # print(A_vecs.size())
        # print("B_vecs Size:")
        # print(B_vecs.size())
        # print("logits Size:")
        # print(logits.size())
        # print("targets Size:")
        # print(target.size())
        batch=logits.size(0)
        dim=A_vecs.size(1)
        mask=-1*torch.ones(batch).to(self.device)
        unlabel_idx=target.eq(mask).int()
        label_idx=target.ne(mask).int()
        
        weight_nce = 0.1
        loss_nce_1 = self.infoNCE_diag(A_vecs, B_vecs,mask=unlabel_idx) * weight_nce
        # loss_nce_2 = self.infoNCE(A_vecs, B_vecs, target) * weight_nce
        loss_nce = loss_nce_1
        loss_main = nn.CrossEntropyLoss()(label_idx.reshape(batch,1)*logits, label_idx*target) * (1 - weight_nce)
        # loss_main=0.
        # print('nce=%s, main=%s, loss=%s' % (loss_nce.detach().cpu().numpy(), loss_main.detach().cpu().numpy(), (loss_nce + loss_main).detach().cpu().numpy()))

        return loss_nce + loss_main   

class Conv1dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv1d.Conv1d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class Conv2dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv2d.Conv2d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class Conv3dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv3d.Conv3d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class CrossDomainTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.pretrain_epochs=self.train_params.get("pre_epochs",80)

    def real_init(self):
        self.use_unlabel=self.train_params.get('use_unlabel',False)
        self.temperature=self.train_params.get('temperature',10)
        self.net=cross_domain.XDCL(self.params).to(self.device)
        self.lr=self.train_params.get('lr',0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,betas=(0.9,0.9))
    
    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            left_data,right_data=do_augment(self.augment,inputs)
            left_data, right_data = [d.to(self.device) for d in [left_data, right_data]]
            outputs = self.get_logits(self.net(left_data, right_data))
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test

    def train(self, train_loader, unlabel_loader,test_loader=None):
        self.temp_unlabel_loader = enumerate(itertools.cycle(unlabel_loader))
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        '''
        预训练，即对比学习部分第一阶段。对model的classifier进行冻结
        '''
        for param in self.net.classifier.parameters():
            param.requires_grad=False
        for epoch in range(self.pretrain_epochs):
            self.net.train()
            epoch_avg_loss.reset()
            self.augment['chosen']=random.randint(0,self.params['data']['spectral_size']-1)
            for i, (unlabel_data, _) in enumerate(unlabel_loader):
                data= unlabel_data.to(self.device)
                '''
                label_data、unlabelled_data大小：
                    batch_size channel height width
                real_target、-1向量大小：
                    batch_size
                '''
                if self.augment:
                # 这里要做的增强，left变成一维光谱，right变为二维平面。并且对二维平面选择的第几层是每个epoch随机选
                    left_data,right_data=do_augment(self.augment,data)
                    left_data, right_data = [d.to(self.device) for d in [left_data, right_data]]
                    outputs = self.net(left_data, right_data)
                else:
                    outputs=self.net(data)
                loss = self.infoNCE_diag(outputs[1],outputs[2],self.temperature)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
        '''
        第二阶段微调，需要将baseEncoder冻结，只训练MLP部分，用ce
        '''
        for param in self.net.classifier.parameters():
            param.requires_grad=True
        for param in self.net.backbone.parameters():
            param.requires_grad=False
        self.augment['ratio']=0
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            self.augment['chosen']=random.randint(0,self.params['data']['spectral_size']-1)
            for i, (label_data, real_target) in enumerate(train_loader):
                data, target= label_data.to(self.device), real_target.to(self.device)
                '''
                不再用unlabel数据，但是还需要aug将光谱、空间提取出来
                '''
                if self.augment:
                # 这里要做的增强，left变成一维光谱，right变为二维平面。并且对二维平面选择的第几层是每个epoch随机选
                    left_data,right_data=do_augment(self.augment,data)
                    left_data, right_data = [d.to(self.device) for d in [left_data, right_data]]
                    outputs = self.net(left_data, right_data)
                else:
                    outputs=self.net(data)
                loss = self.get_loss(outputs, target,0)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
            
        print('Finished Training')
        return True
    

    def infoNCE_diag(self, A_vecs, B_vecs, temperature=10):
        '''
        targets: [batch]  dtype is int
        '''
        # print(A_vecs, B_vecs)
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        matrix_softmax = torch.softmax(matrix_logits, dim=1) # softmax by dim=1
        tempb = matrix_softmax.detach().cpu().numpy()
        # print(np.diag(tempb))
        # print("softmax,", tempb.max(), tempb.min())
        matrix_log = -1 * torch.log(matrix_softmax)
        # here just use dig part
        loss_nce = torch.mean(torch.diag(matrix_log))
        return loss_nce

    def infoNCE(self, A_vecs, B_vecs, targets, temperature=15):
        '''
        targets: [batch]  dtype is int
        '''
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        # tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        matrix_softmax = torch.softmax(matrix_logits, dim=1) # softmax by dim=1
        tempb = matrix_softmax.detach().cpu().numpy()
        # print("softmax,", tempb)
        # print("label,", targets)
        matrix_log = -1 * torch.log(matrix_softmax)

        l = targets.shape[0]
        tb = torch.repeat_interleave(targets.reshape([-1,1]), l, dim=1)
        tc = torch.repeat_interleave(targets.reshape([1,-1]), l, dim=0)
        mask_matrix = tb.eq(tc).int()
        # here just use dig part
        loss_nce = torch.sum(matrix_log * mask_matrix) / torch.sum(mask_matrix)
        return loss_nce

    def get_loss(self, outputs, target,weight_nce=1):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits, A_vecs, B_vecs = outputs
        # print(A_vecs.shape, B_vecs.shape)
        
        loss_nce_1 = self.infoNCE_diag(A_vecs, B_vecs,self.temperature) * weight_nce
        loss_nce_2 = self.infoNCE_diag(B_vecs,A_vecs,self.temperature) * weight_nce
        loss_nce = loss_nce_1+loss_nce_2

        loss_main = nn.CrossEntropyLoss(ignore_index=-1)(logits, target) * (1 - weight_nce)

        print('nce=%s, main=%s, loss=%s' % (loss_nce.detach().cpu().numpy(), loss_main.detach().cpu().numpy(), (loss_nce + loss_main).detach().cpu().numpy()))

        return loss_nce + loss_main   

def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "cross_trainer":
        return CrossTransformerTrainer(params)
    if trainer_type == "conv1d":
        return Conv1dTrainer(params)
    if trainer_type == "conv2d":
        return Conv2dTrainer(params)
    if trainer_type == "conv3d":
        return Conv3dTrainer(params)
    if trainer_type == "svm":
        return SVMTrainer(params) 
    if trainer_type == "random_forest":
        return RandomForestTrainer(params)
    if trainer_type == "knn":
        return KNNTrainer(params)
    if trainer_type == "contra_cross_transformer":
        return ContraCrossTransformerTrainer(params)
    if trainer_type =='cross_domain':
        return CrossDomainTrainer(params)

    assert Exception("Trainer not implemented!")

