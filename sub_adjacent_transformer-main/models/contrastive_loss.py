import random
import torch
import torch.nn.functional as F
import torch.nn as nn

def local_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z1 = crop_z1.view(B ,k,crop_size,D)


    # crop_z2 = z2[:,start:start+crop_leng,:]
    # crop_z2 = crop_z2.view(B ,k,crop_size,D)


    if pooling=='max':
        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2).reshape(B,k,D)

        # crop_z2 = crop_z2.reshape(B*k,crop_size,D)
        # crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        # crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1,2)

    # B X K * K
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k-1, dtype=torch.float32)
    labels = torch.cat([labels,torch.zeros(1,k-1)],0)
    labels = torch.cat([torch.zeros(k,1),labels],-1)

    pos_labels = labels.cuda()
    pos_labels[k-1,k-2]=1.0


    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0,2]=1.0
    neg_labels[-1,-3]=1.0
    neg_labels = neg_labels.cuda()


    similarity_matrix = similarity_matrices[0]

    # select and combine multiple positives
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss



def global_infoNCE(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature)

def InfoNCE(z1, z2, temperature=1.0):

    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss


if __name__ == '__main__':
    x = torch.randn((3, 32, 4)).cuda()
    x_hat = torch.randn((3, 32, 4)).cuda()
    loss1 = local_infoNCE(x, x_hat, k = 16)
    loss2 = global_infoNCE(x, x_hat)
    print(loss1)
    print(loss2)