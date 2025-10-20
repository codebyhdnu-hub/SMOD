from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow_epic import EPICDOMAIN
import torch.nn.functional as F
from scipy import spatial
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

plt.ion()  # Turn on interactive plotting
fig, ax = plt.subplots()
train_acc_list = []
val_acc_list = []
test_acc_list = []
line_train, = ax.plot([], [], label='Train Accuracy')
line_val, = ax.plot([], [], label='Val Accuracy')
line_test, = ax.plot([], [], label='Test Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Train vs Test Accuracy')
ax.legend()
ax.grid(True)


class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04):
        super(LogitNormLoss, self).__init__()
        self.tau = tau

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.tau
        return F.cross_entropy(logit_norm, target)


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature: float):
    T = float(temperature)
    N, C = logits_student.shape
    log_p_s = F.log_softmax(logits_student / T, dim=1)               # (N, C)
    log_p_t = F.log_softmax(logits_teacher.detach() / T, dim=1)      # (N, C)
    y = target.view(-1, 1)
    gt_mask = torch.zeros_like(logits_student, dtype=torch.bool).scatter_(1, y, True)
    not_gt_mask = ~gt_mask

    # -------- TCKD: KL between [p(gt), p(others)] --------
    # log p(gt)
    log_s_gt = log_p_s.gather(1, y)                                   # (N, 1)
    log_t_gt = log_p_t.gather(1, y)                                   # (N, 1)
    # log p(others) = logsumexp over non-GT classes
    neg_inf = -float("inf")
    log_s_others = torch.logsumexp(log_p_s.masked_fill(gt_mask, neg_inf), dim=1, keepdim=True)
    log_t_others = torch.logsumexp(log_p_t.masked_fill(gt_mask, neg_inf), dim=1, keepdim=True)
    # 2-class log-dists
    log_s_2 = torch.cat([log_s_gt, log_s_others], dim=1)              # (N, 2)
    log_t_2 = torch.cat([log_t_gt, log_t_others], dim=1)              # (N, 2)
    tckd_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(log_s_2, log_t_2)

    # -------- NCKD: KL over only non-GT classes (renormalized) --------
    # pick non-GT logits (in log-prob) and renormalize within "others"
    log_s_o = log_p_s.masked_select(not_gt_mask).view(N, C - 1)       # (N, C-1)
    log_t_o = log_p_t.masked_select(not_gt_mask).view(N, C - 1)       # (N, C-1)
    log_s_o = log_s_o - torch.logsumexp(log_s_o, dim=1, keepdim=True)
    log_t_o = log_t_o - torch.logsumexp(log_t_o, dim=1, keepdim=True)
    nckd_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(log_s_o, log_t_o)
    
    return (alpha * tckd_loss + beta * nckd_loss) * (T ** 2)
    
def train_one_step(model, clip, labels, flow, model_flow, epoch_i):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        audio_feat = model_flow.module.backbone.get_feature(flow)
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  
        
    v_feat = model.module.backbone.get_predict(v_feat)
    v_predict, v_emd = model.module.cls_head(v_feat)

    audio_feat = model_flow.module.backbone.get_predict(audio_feat.detach())
    f_predict, f_emd = model_flow.module.cls_head(audio_feat)

    predict = mlp_cls(v_emd, f_emd)

    loss = criterion(predict, labels)
    

    TEMPERATURE = 1.0  # keep same T used in KD
    
    # --- existing logits: v_predict, f_predict; labels: [B] ---
    # Compute mean CE over the batch for each teacher (use logits, not probs)
    v_ce_mean = F.cross_entropy(v_predict / TEMPERATURE, labels, reduction='mean')  # scalar
    f_ce_mean = F.cross_entropy(f_predict / TEMPERATURE, labels, reduction='mean')  # scalar
    
    # Unnormalised weights: exp(-CE)
    v_w_unnorm = torch.exp(-v_ce_mean)
    f_w_unnorm = torch.exp(-f_ce_mean)
    
    # Normalise (across teachers)
    eps = 1e-12
    den = v_w_unnorm + f_w_unnorm + eps
    v_weight = (v_w_unnorm / den)          # scalar
    f_weight = (f_w_unnorm / den)          # scalar
    
    # Apply to logits (broadcast to [B, C])
    teacher_logits = v_weight * v_predict + f_weight * f_predict  # [B, C]

    
    # Compute KD loss
    student_logits = predict
    loss = loss + dkd_loss(student_logits, teacher_logits, labels, 0.2, 0.8, temperature=1.0)

    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()    
    return predict, loss

def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        audio_feat = model_flow.module.backbone.get_feature(flow)  
        audio_feat = model_flow.module.backbone.get_predict(audio_feat)
        f_predict, f_emd = model_flow.module.cls_head(audio_feat)
       
        predict = mlp_cls(v_emd, f_emd)

    loss = criterion(predict, labels)

    return predict, loss


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
       
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/data/scratch/projects/punim1942/multiood_data/EPIC-KITCHENS/',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument("--opt", type=str, default='adam')
    parser.add_argument("--resumef", type=str, default='checkpoint.pt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument('--use_single_pred', action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--a2d_ratio', type=float, default=0.5,
                        help='a2d_ratio')
    parser.add_argument("--sample_number", type=int, default=65)
    parser.add_argument('--ood_entropy_ratio', type=float, default=0.5,
                        help='ood_entropy_ratio')
    parser.add_argument("--start_epoch", type=int, default=10)

    parser.add_argument('--use_a2d', action='store_true')
    parser.add_argument('--use_npmix', action='store_true')
    parser.add_argument('--a2d_max_l1', action='store_true')
    parser.add_argument('--a2d_max_l2', action='store_true')
    parser.add_argument('--a2d_max_hellinger', action='store_true')
    parser.add_argument('--a2d_max_wasserstein', action='store_true')

    parser.add_argument('--max_ood_hellinger', action='store_true')
    parser.add_argument('--max_ood_wasserstein', action='store_true')
    parser.add_argument('--max_ood_l1', action='store_true')
    parser.add_argument('--max_ood_l2', action='store_true')
    parser.add_argument('--a2d_ratio_ood', type=float, default=0.5,
                        help='a2d_ratio_ood')

    parser.add_argument("--nn_k", type=int, default=3)
    parser.add_argument('--mixup_alpha', type=float, default=10.0,
                        help='mixup_alpha')

    parser.add_argument('--logit_norm_tau', type=float, default=0.04,
                        help='logit_norm_tau')
    parser.add_argument('--logit_norm', action='store_true')

    parser.add_argument("--dataset", type=str, default='EPIC')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

    # assign the desired device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    v_dim = 2304
    f_dim = 2048

    num_class = 4

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, device=device,use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    mlp_cls = Encoder(input_dim=v_dim+f_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()

    resume_file = args.resumef
    print("Resuming from ", resume_file)
    checkpoint = torch.load(resume_file)
    # BestTestAcc = checkpoint['BestTestAcc']

    model.load_state_dict(checkpoint['model_state_dict'])
    model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
    # mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])

    model.requires_grad_(False) 
    model_flow.requires_grad_(False) 
    # mlp_cls.requires_grad_(True)
    model.eval()
    model_flow.eval()
    mlp_cls.train()

    
    
    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log_video_flow_%s_near_ood_lr_%s_bsz_%s_%s_%s"%(str(args.dataset), str(args.lr), str(args.bsz), str(args.nepochs), args.opt)

    if args.logit_norm:
        log_name = log_name + '_logit_norm_' + str(args.logit_norm_tau)
    if args.use_single_pred:
        log_name = log_name + '_single_pred'
    
    if args.use_a2d:
        if args.a2d_max_l1:
            log_name = log_name + '_a2d_max_l1_' + str(args.a2d_ratio)
        elif args.a2d_max_l2:
            log_name = log_name + '_a2d_max_l2_' + str(args.a2d_ratio)
        elif args.a2d_max_wasserstein:
            log_name = log_name + '_a2d_max_wasserstein_' + str(args.a2d_ratio)
        elif args.a2d_max_hellinger:
            log_name = log_name + '_a2d_max_hellinger_' + str(args.a2d_ratio)

    if args.use_npmix:
        if args.max_ood_l1:
            log_name = log_name + '_max_ood_l1_' + str(args.a2d_ratio_ood)
        elif args.max_ood_l2:
            log_name = log_name + '_max_ood_l2_' + str(args.a2d_ratio_ood)
        elif args.max_ood_wasserstein:
            log_name = log_name + '_max_ood_wasserstein_' + str(args.a2d_ratio_ood)
        elif args.max_ood_hellinger:
            log_name = log_name + '_max_ood_hellinger_' + str(args.a2d_ratio_ood)

        log_name = log_name + '_entropy_' + str(args.ood_entropy_ratio)
        log_name = log_name + '_start_epoch_' + str(args.start_epoch)
        log_name = log_name + '_nn_k_' + str(args.nn_k) + '_mixup_alpha_' + str(args.mixup_alpha)

    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)
    
    if args.logit_norm:
        criterion = LogitNormLoss(tau=args.logit_norm_tau)
    else:
        criterion = nn.CrossEntropyLoss() 

    criterion = criterion.cuda()
    batch_size = args.bsz

    params = list(mlp_cls.parameters())

    if args.opt == 'adam':
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'sgd':
        optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = CosineAnnealingLR(optim, T_max=args.nepochs)
    
    number_dict = {}
    for i in range(num_class):
        number_dict[i] = 0
    feature_dim = v_dim+f_dim
    data_dict = torch.zeros(num_class, args.sample_number, feature_dim).cuda()
    id_feature_proto = torch.zeros(num_class, feature_dim).cuda()

    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    BestTestAcc = 0
    starting_epoch = 0

    print("starting_epoch: ", starting_epoch)


    train_dataset = EPICDOMAIN(split='train', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)

    test_dataset = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    with open(log_path, "a") as f:
        train_acc_list = []  # NEW
        val_acc_list = []   # NEW 
        test_acc_list = []   # NEW 
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']: #
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                model.eval()
                model_flow.eval()
                mlp_cls.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, spectrogram, labels)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(model, clip, labels, spectrogram, model_flow, epoch_i)
                        else:
                            predict1, loss = validate_one_step(model, clip, labels, spectrogram, model_flow)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                            

                    if split == 'test':
                        currenttestAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestTestAcc = currenttestAcc
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'model_state_dict': model.state_dict(),
                                    'model_flow_state_dict': model_flow.state_dict(),
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()

                                torch.save(save, base_path_model + log_name + '_best.pt')

                        if args.save_checkpoint:
                            save = {
                                'epoch': epoch_i,
                                'BestLoss': BestLoss,
                                'BestEpoch': BestEpoch,
                                'BestAcc': BestAcc,
                                'BestTestAcc': BestTestAcc,
                                'model_state_dict': model.state_dict(),
                                'model_flow_state_dict': model_flow.state_dict(),
                                'optimizer': optim.state_dict(),
                            }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            torch.save(save, base_path_model + log_name + '.pt')
                        
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
                        f.flush()

                if split == 'train':  # NEW
                    train_acc_list.append(acc / float(count))  # NEW
                if split == 'val':  # NEW
                    val_acc_list.append(acc / float(count))  # NEW                    
                if split == 'test':  # NEW
                    test_acc_list.append(acc / float(count))  # NEW
                # Update live plot
                line_train.set_data(range(starting_epoch, starting_epoch + len(train_acc_list)), train_acc_list)
                line_val.set_data(range(starting_epoch, starting_epoch + len(val_acc_list)), val_acc_list)                
                line_test.set_data(range(starting_epoch, starting_epoch + len(test_acc_list)), test_acc_list)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)
                # Save current state of the plot
                plt.savefig(log_name +"train_acc_live.png")   
                

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)

    f.close()
s