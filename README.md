This is my graduation thesis, in which I have made some improvements to FEDFormer.

Including:

1. Added a visualization UI interface for convenient interaction with the model.
2，Introduced a variety of wavelet bases to adapt to different types of data.
3，Utilized wavelet transform to assign weights in the frequency domain. (This method can slightly improve the accuracy in short-term prediction but increases the computation time.)
4，Change the sliding average time series segmentation to wavelet transform-based sliding average segmentation.

Two new libraries are needed.

pip install pytorch-wavelets
pip install PyWavelets

For detailed information, refer to “FEDformer: Frequency Enhanced Decomposed Transformer for Long - term Series Forecasting” 
