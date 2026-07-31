# import library
import torch
import torch.nn as nn

class DFL(nn.Module):
    
        # initialize
        def __init__(
            self,
            reg_max
        ):
            
            # nn.Module reset to use PyTorch
            super(DFL, self).__init__()
            
            # Integral Conv
            # Convert probability distribution into expected distance
            self.conv = nn.Conv2d(
                in_channels=reg_max,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

            # Integral weights: [0, 1, 2, ..., reg_max - 1]
            projection = torch.arange(
                reg_max,
                dtype=torch.float32
            )

            projection = projection.reshape(
                1,
                reg_max,
                1,
                1
            )

            # Set fixed convolution weights
            self.conv.weight.data.copy_(projection)

            # DFL integral weights are not trainable
            self.conv.requires_grad_(False)

        # forward
        def forward(self, x):

            # input shape
            # x: [batch_size, 4 * reg_max, num_anchors]
            batch_size, channels, num_anchors = x.shape
            
            # Separate four bounding-box directions
            # [B, 4 * reg_max, A]
            #             ↓
            # [B, 4, reg_max, A]
            x = x.reshape(
                batch_size,
                4,
                self.reg_max,
                num_anchors
            )

            # Move distribution dimension to Conv2d channel dimension
            # [B, 4, reg_max, A]
            #             ↓
            # [B, reg_max, 4, A]
            x = x.transpose(1, 2)

            # Convert logits into probability distributions
            x = torch.softmax(
                x,
                dim=1
            )

            # Calculate expected distances
            # 0*p0 + 1*p1 + ... + (reg_max-1)*p_last
            #
            # [B, reg_max, 4, A]
            #             ↓
            # [B, 1, 4, A]
            x = self.conv(x)

            # Remove unnecessary channel dimension
            # [B, 1, 4, A]
            #             ↓
            # [B, 4, A]
            x = x.reshape(
                batch_size,
                4,
                num_anchors
            )

            # return
            return x