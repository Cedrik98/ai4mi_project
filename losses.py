#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        self.smooth = kwargs['smooth']

    def __call__(self, pred_probs, target):
        assert pred_probs.shape == target.shape
        
        size = pred_probs.size(0)
        pred_probs = pred_probs.reshape(size, -1)
        target_ = target.reshape(size, -1)

        intersection = pred_probs * target_
        dice_score = (2 * intersection.sum(1) + self.smooth) / (pred_probs.sum(1) + target_.sum(1) + self.smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class FocalLoss():
    def __init__(self, **kwargs):
        self.alpha = kwargs['alpha']
        self.gamma = kwargs['gamma']
        self.idk = kwargs['idk']

    def __call__(self, pred_probs, target, pred_seg):
        assert pred_probs.shape == target.shape
        
        p_t = (pred_probs * target).sum(dim=1)  

        # log(p_t), cross-entropy 
        log_p_t = (p_t + 1e-10).log()

        # modulating factor
        modulating_factor = (1 - p_t) ** self.gamma

        focal_loss = -self.alpha * modulating_factor * log_p_t

        return focal_loss.mean()


# Class to combine CrossEntropy and Dice Loss
class CEDiceLoss():
    def __init__(self, **kwargs):
        self.dice_loss_func = DiceLoss(**kwargs)
        self.ce_loss_func = CrossEntropy(**kwargs)
        self.dice_weight = kwargs['dice_weight']
        self.ce_weight = kwargs['ce_weight']

    def __call__(self, pred_probs, target):
        # Calculate Dice loss
        dice_loss = self.dice_loss_func(pred_probs, target)

        # Calculate Cross-Entropy loss
        ce_loss = self.ce_loss_func(pred_probs, target)

        # Combine the two losses
        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return combined_loss


# Class to combine Focal Loss and Dice Loss alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5, smooth=1, idk=[0, 1, 2, 3, 4],
class FocalDiceLoss():
    def __init__(self, **kwargs):
        self.dice_loss_func = DiceLoss(**kwargs)
        self.focal_loss_func = FocalLoss(**kwargs)
        self.dice_weight = kwargs['dice_weight']
        self.focal_weight = kwargs['focal_weight']

    def __call__(self, pred_probs, target, pred_seg):
        # Calculate Dice loss
        dice_loss = self.dice_loss_func(pred_probs, target)

        # Calculate Focal loss
        focal_loss = self.focal_loss_func(pred_probs, target, pred_seg)

        # Combine the two losses
        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return combined_loss