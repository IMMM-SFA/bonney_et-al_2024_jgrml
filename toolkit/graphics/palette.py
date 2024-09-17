import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style()

### set rcParams
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (10,6)

# font sizes
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 12
# plt.rcParams['axes.titlesize'] = 'large'

SECTOR_COLORS = {
    "IRR": "#4c7d0d",
    "MUN": "#006490",
    "MIN": "#888888",
    "IND": "#7d0d7c",
    "POW": "#664e00", 
}

DROUGHT_PALETTE = sns.color_palette("rocket")
DROUGHT_PALETTE.reverse()

MEANPROPS={"marker": "o",
            "markeredgecolor": "black",
            "markerfacecolor": "white",
            "markersize": "10"}

rcparams = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

plt.rcParams.update(rcparams)
