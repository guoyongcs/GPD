import torch
import torch.nn as nn
import torch.nn.functional as F


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y, x


class ReviewKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, mid_channel, expand_ratio, custom_shape=None
    ):
        super(ReviewKD, self).__init__()
        self.shapes = custom_shape if custom_shape is not None else [1,7,14,28,56]
        self.student = student
        self.expand_ratio = expand_ratio
        self.in_channels = in_channels

        abfs = nn.ModuleList()
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        self.abfs = abfs[::-1]

        if expand_ratio > 1:
            abfs_dual = nn.ModuleList()
            for idx, in_channel in enumerate(in_channels):
                last_module = idx < len(in_channels) - 1
                abfs_dual.append(ABF(in_channel * expand_ratio, mid_channel, out_channels[idx], last_module))
            self.abfs_dual = abfs_dual[::-1]
        else:
            self.abfs_dual = None

    def forward(self, x):
        student_features = self.student(x, is_feat=True)
        logit = student_features[1]
        x = student_features[0][::-1]
        results = []
        in_channel = x[0].shape[1]
        if in_channel == self.in_channels[-1]:
            active_abfs = self.abfs
        else:
            active_abfs = self.abfs_dual
        out_features, res_features = active_abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], active_abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return results, logit


def build_review_kd(arch, model, expand_ratio=1):
    if arch == 'resnet18':
        in_channels = [64,128,256,512,512]
        out_channels = [64,128,256,512,512]
        mid_channel = 512
        custom_shape = None
    elif arch == 'mobilenet' or arch == 'MobileNet':
        in_channels = [128,256,512,1024,1024]
        out_channels = [256,512,1024,2048,2048]
        mid_channel = 256
        custom_shape = None
    elif arch == 'fan_tiny_8_p4_hybrid':
        in_channels = [192,192,192,192,192]
        out_channels = [384,384,384,384,384]
        mid_channel = 256
        custom_shape = [14, 14, 14, 14, 14]
    elif arch == 'rvt_tiny':
        in_channels = [192,192,192,192,192]
        out_channels = [384,384,384,384,384]
        mid_channel = 256
        custom_shape = [14, 14, 14, 14, 14]
    else:
        print(arch, 'is not defined.')
        assert False
    model = ReviewKD(model, in_channels, out_channels, mid_channel, expand_ratio, custom_shape=custom_shape)
    return model


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all