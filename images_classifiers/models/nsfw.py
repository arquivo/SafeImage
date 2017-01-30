from kaffe.tensorflow import Network

class ResNet_50_1by2_nsfw(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, relu=False, name='conv_1')
             .batch_normalization(relu=True, name='bn_1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage0_block0_proj_shortcut')
             .batch_normalization(name='bn_stage0_block0_proj_shortcut'))

        (self.feed('pool1')
             .conv(1, 1, 32, 1, 1, relu=False, name='conv_stage0_block0_branch2a')
             .batch_normalization(relu=True, name='bn_stage0_block0_branch2a')
             .conv(3, 3, 32, 1, 1, relu=False, name='conv_stage0_block0_branch2b')
             .batch_normalization(relu=True, name='bn_stage0_block0_branch2b')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage0_block0_branch2c')
             .batch_normalization(name='bn_stage0_block0_branch2c'))

        (self.feed('bn_stage0_block0_proj_shortcut', 
                   'bn_stage0_block0_branch2c')
             .add(name='eltwise_stage0_block0')
             .relu(name='relu_stage0_block0')
             .conv(1, 1, 32, 1, 1, relu=False, name='conv_stage0_block1_branch2a')
             .batch_normalization(relu=True, name='bn_stage0_block1_branch2a')
             .conv(3, 3, 32, 1, 1, relu=False, name='conv_stage0_block1_branch2b')
             .batch_normalization(relu=True, name='bn_stage0_block1_branch2b')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage0_block1_branch2c')
             .batch_normalization(name='bn_stage0_block1_branch2c'))

        (self.feed('relu_stage0_block0', 
                   'bn_stage0_block1_branch2c')
             .add(name='eltwise_stage0_block1')
             .relu(name='relu_stage0_block1')
             .conv(1, 1, 32, 1, 1, relu=False, name='conv_stage0_block2_branch2a')
             .batch_normalization(relu=True, name='bn_stage0_block2_branch2a')
             .conv(3, 3, 32, 1, 1, relu=False, name='conv_stage0_block2_branch2b')
             .batch_normalization(relu=True, name='bn_stage0_block2_branch2b')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage0_block2_branch2c')
             .batch_normalization(name='bn_stage0_block2_branch2c'))

        (self.feed('relu_stage0_block1', 
                   'bn_stage0_block2_branch2c')
             .add(name='eltwise_stage0_block2')
             .relu(name='relu_stage0_block2')
             .conv(1, 1, 256, 2, 2, relu=False, name='conv_stage1_block0_proj_shortcut')
             .batch_normalization(name='bn_stage1_block0_proj_shortcut'))

        (self.feed('relu_stage0_block2')
             .conv(1, 1, 64, 2, 2, relu=False, name='conv_stage1_block0_branch2a')
             .batch_normalization(relu=True, name='bn_stage1_block0_branch2a')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv_stage1_block0_branch2b')
             .batch_normalization(relu=True, name='bn_stage1_block0_branch2b')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv_stage1_block0_branch2c')
             .batch_normalization(name='bn_stage1_block0_branch2c'))

        (self.feed('bn_stage1_block0_proj_shortcut', 
                   'bn_stage1_block0_branch2c')
             .add(name='eltwise_stage1_block0')
             .relu(name='relu_stage1_block0')
             .conv(1, 1, 64, 1, 1, relu=False, name='conv_stage1_block1_branch2a')
             .batch_normalization(relu=True, name='bn_stage1_block1_branch2a')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv_stage1_block1_branch2b')
             .batch_normalization(relu=True, name='bn_stage1_block1_branch2b')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv_stage1_block1_branch2c')
             .batch_normalization(name='bn_stage1_block1_branch2c'))

        (self.feed('relu_stage1_block0', 
                   'bn_stage1_block1_branch2c')
             .add(name='eltwise_stage1_block1')
             .relu(name='relu_stage1_block1')
             .conv(1, 1, 64, 1, 1, relu=False, name='conv_stage1_block2_branch2a')
             .batch_normalization(relu=True, name='bn_stage1_block2_branch2a')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv_stage1_block2_branch2b')
             .batch_normalization(relu=True, name='bn_stage1_block2_branch2b')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv_stage1_block2_branch2c')
             .batch_normalization(name='bn_stage1_block2_branch2c'))

        (self.feed('relu_stage1_block1', 
                   'bn_stage1_block2_branch2c')
             .add(name='eltwise_stage1_block2')
             .relu(name='relu_stage1_block2')
             .conv(1, 1, 64, 1, 1, relu=False, name='conv_stage1_block3_branch2a')
             .batch_normalization(relu=True, name='bn_stage1_block3_branch2a')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv_stage1_block3_branch2b')
             .batch_normalization(relu=True, name='bn_stage1_block3_branch2b')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv_stage1_block3_branch2c')
             .batch_normalization(name='bn_stage1_block3_branch2c'))

        (self.feed('relu_stage1_block2', 
                   'bn_stage1_block3_branch2c')
             .add(name='eltwise_stage1_block3')
             .relu(name='relu_stage1_block3')
             .conv(1, 1, 512, 2, 2, relu=False, name='conv_stage2_block0_proj_shortcut')
             .batch_normalization(name='bn_stage2_block0_proj_shortcut'))

        (self.feed('relu_stage1_block3')
             .conv(1, 1, 128, 2, 2, relu=False, name='conv_stage2_block0_branch2a')
             .batch_normalization(relu=True, name='bn_stage2_block0_branch2a')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv_stage2_block0_branch2b')
             .batch_normalization(relu=True, name='bn_stage2_block0_branch2b')
             .conv(1, 1, 512, 1, 1, relu=False, name='conv_stage2_block0_branch2c')
             .batch_normalization(name='bn_stage2_block0_branch2c'))

        (self.feed('bn_stage2_block0_proj_shortcut', 
                   'bn_stage2_block0_branch2c')
             .add(name='eltwise_stage2_block0')
             .relu(name='relu_stage2_block0')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage2_block1_branch2a')
             .batch_normalization(relu=True, name='bn_stage2_block1_branch2a')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv_stage2_block1_branch2b')
             .batch_normalization(relu=True, name='bn_stage2_block1_branch2b')
             .conv(1, 1, 512, 1, 1, relu=False, name='conv_stage2_block1_branch2c')
             .batch_normalization(name='bn_stage2_block1_branch2c'))

        (self.feed('relu_stage2_block0', 
                   'bn_stage2_block1_branch2c')
             .add(name='eltwise_stage2_block1')
             .relu(name='relu_stage2_block1')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage2_block2_branch2a')
             .batch_normalization(relu=True, name='bn_stage2_block2_branch2a')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv_stage2_block2_branch2b')
             .batch_normalization(relu=True, name='bn_stage2_block2_branch2b')
             .conv(1, 1, 512, 1, 1, relu=False, name='conv_stage2_block2_branch2c')
             .batch_normalization(name='bn_stage2_block2_branch2c'))

        (self.feed('relu_stage2_block1', 
                   'bn_stage2_block2_branch2c')
             .add(name='eltwise_stage2_block2')
             .relu(name='relu_stage2_block2')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage2_block3_branch2a')
             .batch_normalization(relu=True, name='bn_stage2_block3_branch2a')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv_stage2_block3_branch2b')
             .batch_normalization(relu=True, name='bn_stage2_block3_branch2b')
             .conv(1, 1, 512, 1, 1, relu=False, name='conv_stage2_block3_branch2c')
             .batch_normalization(name='bn_stage2_block3_branch2c'))

        (self.feed('relu_stage2_block2', 
                   'bn_stage2_block3_branch2c')
             .add(name='eltwise_stage2_block3')
             .relu(name='relu_stage2_block3')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage2_block4_branch2a')
             .batch_normalization(relu=True, name='bn_stage2_block4_branch2a')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv_stage2_block4_branch2b')
             .batch_normalization(relu=True, name='bn_stage2_block4_branch2b')
             .conv(1, 1, 512, 1, 1, relu=False, name='conv_stage2_block4_branch2c')
             .batch_normalization(name='bn_stage2_block4_branch2c'))

        (self.feed('relu_stage2_block3', 
                   'bn_stage2_block4_branch2c')
             .add(name='eltwise_stage2_block4')
             .relu(name='relu_stage2_block4')
             .conv(1, 1, 128, 1, 1, relu=False, name='conv_stage2_block5_branch2a')
             .batch_normalization(relu=True, name='bn_stage2_block5_branch2a')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv_stage2_block5_branch2b')
             .batch_normalization(relu=True, name='bn_stage2_block5_branch2b')
             .conv(1, 1, 512, 1, 1, relu=False, name='conv_stage2_block5_branch2c')
             .batch_normalization(name='bn_stage2_block5_branch2c'))

        (self.feed('relu_stage2_block4', 
                   'bn_stage2_block5_branch2c')
             .add(name='eltwise_stage2_block5')
             .relu(name='relu_stage2_block5')
             .conv(1, 1, 1024, 2, 2, relu=False, name='conv_stage3_block0_proj_shortcut')
             .batch_normalization(name='bn_stage3_block0_proj_shortcut'))

        (self.feed('relu_stage2_block5')
             .conv(1, 1, 256, 2, 2, relu=False, name='conv_stage3_block0_branch2a')
             .batch_normalization(relu=True, name='bn_stage3_block0_branch2a')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv_stage3_block0_branch2b')
             .batch_normalization(relu=True, name='bn_stage3_block0_branch2b')
             .conv(1, 1, 1024, 1, 1, relu=False, name='conv_stage3_block0_branch2c')
             .batch_normalization(name='bn_stage3_block0_branch2c'))

        (self.feed('bn_stage3_block0_proj_shortcut', 
                   'bn_stage3_block0_branch2c')
             .add(name='eltwise_stage3_block0')
             .relu(name='relu_stage3_block0')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv_stage3_block1_branch2a')
             .batch_normalization(relu=True, name='bn_stage3_block1_branch2a')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv_stage3_block1_branch2b')
             .batch_normalization(relu=True, name='bn_stage3_block1_branch2b')
             .conv(1, 1, 1024, 1, 1, relu=False, name='conv_stage3_block1_branch2c')
             .batch_normalization(name='bn_stage3_block1_branch2c'))

        (self.feed('relu_stage3_block0', 
                   'bn_stage3_block1_branch2c')
             .add(name='eltwise_stage3_block1')
             .relu(name='relu_stage3_block1')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv_stage3_block2_branch2a')
             .batch_normalization(relu=True, name='bn_stage3_block2_branch2a')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv_stage3_block2_branch2b')
             .batch_normalization(relu=True, name='bn_stage3_block2_branch2b')
             .conv(1, 1, 1024, 1, 1, relu=False, name='conv_stage3_block2_branch2c')
             .batch_normalization(name='bn_stage3_block2_branch2c'))

        (self.feed('relu_stage3_block1', 
                   'bn_stage3_block2_branch2c')
             .add(name='eltwise_stage3_block2')
             .relu(name='relu_stage3_block2')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='pool')
             .fc(2, relu=False, name='fc_nsfw')
             .softmax(name='prob'))