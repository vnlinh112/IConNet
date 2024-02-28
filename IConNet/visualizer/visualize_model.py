from ..firconv.signal import general_cosine_window, get_window_freq_response
from .visualize_signal import visualize_window
import torch
from typing import Literal, Optional
from einops import repeat, rearrange
import numpy as np
from . import DEFAULT_SAMPLE_RATE, display_module
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import math
import seaborn as sns
sns.set_theme(style="dark")

def count_parameters(model, all=False):
    if all:
        params_list = [p.numel() for p in model.parameters() if p.requires_grad]
    else:
        params_list = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    return sum(params_list)

def print_parameters(model, all=False):
    if all:
        params = model.parameters()
    else:
        params = model.named_parameters()
    [print(p) for p in params]

def visualize_model(
        model, input_shape, device='cpu',
        classes_to_visit={}):
    print(model)
    model.to(device)
    input = torch.rand(input_shape).to(device)
    display_module(
        model, input, 
        classes_to_visit=classes_to_visit)


def visualize_multilines_by_group(
        data: pd.DataFrame, 
        x_colname: str, 
        y_colname: str, 
        group_by_colname: str, 
        subchart_value_colname: str, 
        subchart_variance_colname: str,
        linestyle_by_colname: str, 
        sample_rate=1,
        title: str="", 
        xticks = None,
        xhighlight: Optional[float] = None,
        xhighlight_color: str = 'red'):
    
    print(title)

    g = sns.relplot(
        data=data, x=x_colname, y=y_colname, 
        col=group_by_colname, hue=group_by_colname, 
        col_wrap=2, legend=False, zorder=5,
        kind="line", palette="crest", linewidth=2, 
        height=2, aspect=1.5 
    )

    for group, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        sub_df = data[data[group_by_colname]==group]
        value_mean = int(sub_df[subchart_value_colname].mean() * sample_rate)
        value_var = int(sub_df[subchart_variance_colname].mean() * sample_rate)
        subtitle = f"{value_mean} ± {value_var} Hz"
        ax.text(.3, 1, subtitle, transform=ax.transAxes, fontweight="bold")
        if xhighlight:
            ax.axhline(xhighlight, color=xhighlight_color, zorder=1)
        # Plot every window in the background
        sns.lineplot(
            data=sub_df, x=x_colname, y=y_colname, 
            units=group_by_colname, 
            style=linestyle_by_colname, legend=False,
            estimator=None, color=".7", linewidth=1, ax=ax
        )
    if xticks is not None:
        ax.set_xticks(xticks)
    g.set_titles("")
    g.tight_layout()


class FilterVisualizer():
    def __init__(self, 
                lowcut_bands, bandwidths,
                window_params=[0.5, 0.5], 
                window_length=64,
                sample_rate=DEFAULT_SAMPLE_RATE):
        if type(lowcut_bands) == torch.Tensor or type(lowcut_bands) == torch.nn.parameter.Parameter:
            lowcut_bands = lowcut_bands.detach().cpu().numpy()
        self.lowcut_bands = lowcut_bands
        if type(bandwidths) == torch.Tensor or type(bandwidths) == torch.nn.parameter.Parameter:
            bandwidths = bandwidths.detach().cpu().numpy()
        self.bandwidths = bandwidths
        if type(window_params) == torch.Tensor or type(window_params) == torch.nn.parameter.Parameter:
            window_params = window_params.detach().cpu().numpy()
        self.window_params = window_params # (out_channels, in_channels, n_params)
        self.window_length = window_length

        self.sample_rate = sample_rate
        self.filters = None
        self.filters_time = None
        self.filters_freq = None
        self.visualize_filters_by_group = visualize_multilines_by_group

    def visualize_one_window(self, n_out=0, n_inp=0, 
                             window_length=128,
                     f_xhighlight=-20, f_ylim=[-100,10]):
        assert len(self.window_params.shape) == 3
        window_params = self.window_params[n_out][n_inp]
        window = general_cosine_window(
            window_length, 
            window_params, 
            out_type=np.ndarray)
        visualize_window(window, 
                        f_xhighlight=f_xhighlight, 
                        f_ylim=f_ylim)

    def visualize_window_params(self, dim=None):
        if dim is None:
            dim = self.window_params.shape[-1]
        dim = np.clip(dim, a_max=3, a_min=2)
        w1x = self.window_params[..., 0].flatten()
        w1y = self.window_params[..., 1].flatten()
        w1c = w1x + w1y
        if dim == 2:
            plt.scatter(w1x, w1y, c=w1c)
        else:
            fig = plt.figure(figsize=(12,6))
            ax = plt.axes(projection ='3d')
            w1z = self.window_params[...,2].flatten()
            w1c = w1x + w1y
            ax.set_xlim([w1x.min() - 0.1, w1x.max() + 0.1])
            ax.set_ylim([w1y.min() - 0.1, w1y.max() + 0.1])
            ax.set_zlim([w1z.min() - 0.1, w1z.max() + 0.1])
            ax.scatter(w1x, w1y, w1z, c=w1c, edgecolor='white')

    def visualize_bands(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        axes[0].hist(self.lowcut_bands.flatten())
        axes[1].hist(self.bandwidths.flatten())

    def _get_freq_bins(self, x, 
                       window_length, 
                       x_n_bin, y_n_bin):
        x['lowcut_bin'] = max(math.ceil(
                x['lowcut'] / self.lowcut_bands.max() * y_n_bin) - 1, 0)
        x['bandwidth_bin'] = max(math.ceil(
                x['bandwidth'] / self.bandwidths.max() * x_n_bin) - 1, 0)
        x['window'] = general_cosine_window(
                window_length, x['window_params']).numpy().flatten()
                
        x['order'] = (y_n_bin - 1 - x['lowcut_bin']) * x_n_bin + x['bandwidth_bin']
        return x

    def create_filters(
            self, window_length, x_n_bin, y_n_bin):
        
        self.filters = pd.DataFrame(data={
            'lowcut': self.lowcut_bands.flatten(), 
            'bandwidth': self.bandwidths.flatten(), 
            'window_params': list(rearrange(
                self.window_params, 'h c p -> (h c) p')) 
        })

        self.filters = self.filters.apply(
            lambda x: self._get_freq_bins(
                x, window_length, x_n_bin, y_n_bin), axis=1)
        
        
    def create_filters_time(self, window_length):
        h, c = self.lowcut_bands.shape
        self.filters_time = pd.DataFrame(data={
            'lowcut': repeat(self.lowcut_bands, 
                             'h c -> (h c k)', k=window_length), 
            'bandwidth': repeat(self.bandwidths, 
                                'h c -> (h c k)', k=window_length),  
            'window': np.hstack(self.filters['window'].values),
            'window_t': repeat(np.arange(window_length), 
                               'k -> (h c k)', h=h, c=c, k=window_length),
            'lowcut_bin': repeat(self.filters['lowcut_bin'].values, 
                                 '(h c) -> (h c k)', h=h, c=c, k=window_length),
            'bandwidth_bin': repeat(self.filters['bandwidth_bin'].values, 
                                    '(h c) -> (h c k)', h=h, c=c, k=window_length),
            'order': repeat(self.filters['order'].values, 
                            '(h c) -> (h c k)', h=h, c=c, k=window_length), 
            'win_id': repeat(np.arange(h*c), 
                             '(h c)-> (h c k)', h=h, c=c, k=window_length)
        })

    def create_filters_freq(self, window_length):
        full_window_response_length = window_length*10
        window_response_length = full_window_response_length
        self.filters['window_response'] = self.filters['window'].apply(
            lambda x: get_window_freq_response(x)[:window_response_length])
        freq = np.fft.fftfreq(
            full_window_response_length,d=1/2)[:window_response_length]
        
        h, c = self.lowcut_bands.shape
        self.filters_freq = pd.DataFrame(data={
            'lowcut': repeat(
                self.lowcut_bands, 
                'h c -> (h c k)', k=window_response_length), 
            'bandwidth': repeat(
                self.bandwidths, 
                'h c -> (h c k)', k=window_response_length),  
            'window_response': np.hstack(
                self.filters['window_response'].values),
            'window_f': repeat(
                freq, 'k -> (h c k)', h=h, c=c, k=window_response_length),
            'lowcut_bin': repeat(
                self.filters['lowcut_bin'].values, 
                '(h c) -> (h c k)', h=h, c=c, k=window_response_length),
            'bandwidth_bin': repeat(
                self.filters['bandwidth_bin'].values, 
                '(h c) -> (h c k)', h=h, c=c, k=window_response_length),
            'order': repeat(
                self.filters['order'].values, 
                '(h c) -> (h c k)', h=h, c=c, k=window_response_length), 
            'win_id': repeat(
                np.arange(h*c), 
                '(h c)-> (h c k)', h=h, c=c, k=window_response_length)
        })

    def visualize_filters(
            self, filter_domain: Literal['time', 'freq'],
            window_length=17, x_n_bin=3, y_n_bin=5):
        
        if window_length == None:
            window_length = self.window_length
        
        if self.filters is None:
            self.create_filters(window_length, x_n_bin, y_n_bin)
        
        if self.filters_time is None:
            self.create_filters_time(window_length)
        
        if filter_domain == 'freq':
            if self.filters_freq is None:
                self.create_filters_freq(window_length)
            self.display_filters_in_freq(window_length)
        else:
            self.display_filters_in_time(window_length)

    def display_filters_in_time(self, window_length):
        domain = "Time domain"
        title = f"{domain}: Filters' shapes vary by frequency bins"
        self.visualize_filters_by_group(
            data=self.filters_time, 
            x_colname="window_t", 
            y_colname="window", 
            group_by_colname="order", 
            subchart_value_colname="lowcut", 
            subchart_variance_colname="bandwidth",
            linestyle_by_colname="win_id", 
            sample_rate=self.sample_rate,
            title=title,
            xticks=np.linspace(0, window_length, 3),
            xhighlight=None
        )

    def display_filters_in_freq(self, window_length):
        domain = "Frequency domain"
        title = f"{domain}: Filters' shapes vary by frequency bins"
        self.visualize_filters_by_group(
            data=self.filters_freq, 
            x_colname="window_f", 
            y_colname="window_response", 
            group_by_colname="order", 
            subchart_value_colname="lowcut", 
            subchart_variance_colname="bandwidth",
            linestyle_by_colname="win_id", 
            sample_rate=self.sample_rate,
            title=title,
            xhighlight=-20.
        )


