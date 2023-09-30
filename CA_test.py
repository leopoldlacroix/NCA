# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import cv2 

# %%

def inverse_gaussian(x):
    return -1./tf.math.pow(2., (0.6*tf.math.pow(x, 2.)))+1.

def circular_padding(inputs):
    top__bottom_cropped = tf.keras.layers.Cropping2D(cropping=((0, inputs.shape[1]-1), (0, 0)))(inputs)
    bottom__top_cropped = tf.keras.layers.Cropping2D(cropping=((inputs.shape[1]-1, 0), (0, 0)))(inputs)
    circular_1 = tf.keras.layers.Concatenate(axis=1)([bottom__top_cropped, inputs, top__bottom_cropped])
    
    
    right__left_cropped = tf.keras.layers.Cropping2D(cropping=((0, 0), (inputs.shape[1]-1, 0)))(circular_1) #top, bottom, left, right
    left__right_cropped = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, inputs.shape[1]-1)))(circular_1) #top, bottom, left, right
    circular_ = tf.keras.layers.Concatenate(axis=2)([right__left_cropped, circular_1, left__right_cropped])
    return circular_

class ConvCA(tf.keras.Model):
    """Applies a convolutional filter over an image and passes the result through
    an activation function."""
    def __init__(self, filter_size=3, filter_scale=0.5, act=tf.tanh, n_chanels=3):
        """Sets up the filter and activation function.
        Defaults to:
        - filter_size=3, 
        - filter_scale=0.5, # sometimes it's nice to scale down the filter
        - act=torch.tanh
        """
        super().__init__()
        # shape: h, w, in_channels, out_channels
        self.filter = tf.random.normal((3, 3, 3, n_chanels)) # h, w, in_channels, out_channels
        self.act = act

    def call(self, inputs):
        cinputs = circular_padding(inputs)
        conv_out_1 = tf.nn.conv2d(input = cinputs, filters=self.filter, strides=[1,1,1,1], padding='VALID')
        # print(inputs, conc_axis_2)
        return self.act(conv_out_1)
  
def get_plot(heatmaps: list[go.Heatmap], fps):
        
    fig = go.Figure(
        data=[heatmaps[0]],
        layout=go.Layout(
            title="Frame 0",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {'frame':{'duration': fps}, }]
                        )]
            )],
        ),
        frames=[
            go.Frame(
                data=[heatmaps[i]],
                layout=go.Layout(title_text=f"Frame {i}")
            )
            for i in range(1, len(heatmaps))
        ],
    )
    
    return fig

def render(ca: ConvCA, grid, n=1000, save_every=1, exp_frac=0, fps=10):
    """ Render n steps of a ca starting from a random grid.
    Saves an image every save_every steps into the steps/ folder.
    Smooths the aniation with exponential averaging set by exp_frac
    """
    def get_array_from_grid(grid):
        return tf.clip_by_value(grid[0,:,:,:3], 0, 1).numpy()    
    
    out = cv2.VideoWriter('video.mp4' ,cv2.VideoWriter_fourcc(*'DIVX'), 15, grid.shape[1:3])
    heatmaps = []
    array_i = get_array_from_grid(grid)
    for i in range(n):
        if i % save_every == 0:
            array_i = exp_frac*array_i + (1-exp_frac)*get_array_from_grid(grid)
            heatmaps.append(go.Image(z=array_i*255))
            out.write(array_i)
        grid = ca(grid)

    out.release()
    cv2.destroyAllWindows()
    return get_plot(heatmaps, fps=fps)

# %%
shape = (100,100)
batchs, n_ch = 1, 5
ca = ConvCA(filter_scale=0.5, act = inverse_gaussian)
ca.filter = np.stack([np.expand_dims([[0.68, -0.9, 0.68],[-0.9, -0.66, -0.9],[0.68, -0.9, 0.68]], axis = -1)]*n_ch, axis=-1)
# x = np.array([np.stack([np.arange(np.prod(shape)).reshape(shape)]*n_ch, axis = -1)]).astype(float)/100
# out = ca(x)

grid = tf.concat(
    [tf.stack([tf.random.uniform((1, *shape), minval=0, maxval=1)]*(n_ch), axis = -1)]*2,
    axis = 0
)

fig = render(ca, grid, n = 100, save_every=2)
fig.show()
# %%
# hard coded filter 3x3 x 4 channels
filters = np.stack([
     np.array([[0]*3, [0,1,0], [0]*3]),
     np.array([[-1,0,1], np.array([-1,0,1])*2, [-1,0,1]]), 
     np.array([[-1,0,1], np.array([-1,0,1])*2, [-1,0,1]]).T, 
     np.array([[1,2,1], [2,-12,2], [1,2,1]])],
axis = -1)[:,:,None]

class SimpleCA(tf.keras.Model):
    def __init__(self, hidden_n=6, zero_w2=True):
        super().__init__()
        # shape: h, w, in_channels, out_channels
        self.filters = tf.convert_to_tensor(filters)
        self.chn = 4
        
        self.w1 = tf.keras.layers.Conv2D(hidden_n, 1)
        self.relu = tf.keras.layers.ReLU()
        self.w2 = tf.keras.layers.Conv2D(4, 1, use_bias=False, filter_initializer= "zeros"if zero_w2 else "glorot_uniform")
    
    def prechannel_conv(self, inputs):
        cinputs = circular_padding(inputs)
        cconv_out_1 = tf.nn.conv2d(input = cinputs, filters=self.filters, strides=[1,1,1,1], padding='VALID')
        return cconv_out_1
        
    def call(self, inputs, update_rate):
        cconv_1 = self.prechannel_conv(inputs)
        w1_out = self.w1(cconv_1)
        relu1_out = self.relu(w1_out)
        w2_out = self.w2(relu1_out)
        b, h, w, c = w2_out.shape
        # The update mask means that only some pixels are updated:
        update_mask = np.floor(np.random.rand((b, h, w, 1))+update_rate)
        
        return inputs+w2_out*update_mask
    
    def to_rgb(self, inputs):
        return inputs[:,:,:,:3]
    
    def seed(self, n, sz=128):
        """Initializes n 'grids', size sz. In this case all 0s."""
        return np.zeros((n, sz, sz, self.chn))
      
ca = SimpleCA(8, zero_w2=False) 
grid = ca.seed(1)

render(ca, grid, n=100)
# %%
