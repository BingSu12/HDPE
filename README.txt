############################################################################
# # # Hierarchical Dynamic Parsing and Encoding for Action Recognition # # # 
############################################################################

1. Introduction.

This package includes the propotype MATLAB codes for performing the hierarchical dynamic parsing and encoding, described in

	Bing Su, Jiahuan Zhou, Xiaoqing Ding, and Ying Wu, "Unsupervised Hierarchical Dynamic Parsing and Encoding for Action Recognition ", IEEE Trans. on Image Processing (TIP), 2017, 26(12), pp. 5784-5799. DOI: 10.1109/TIP.2017.2745212

	Bing Su, Jiahuan Zhou, Xiaoqing Ding, Hao Wang, and Ying Wu, "Hierarchical Dynamic Parsing and Encoding for Action Recognition", Proc. European Conf. on Computer Vision (ECCV), 2016, pp. 202-217.

----------------------------------------------------------------------

Tested under windows 7 x64, matlab R2015b.

############################################################################

2. License & disclaimer.

    The codes can be used for research purposes only. This package is strictly for non-commercial academic use only.

############################################################################

3.  Usage & Dependency.

This package shows how to perform unsupervised hierarchical dynamic parsing and encoding (HDPE) for action recognition on the ChaLearn dataset in Matlab R2015b. HDPE encodes a sequence into a vector representation.
  
- ChaLearn_HDPE.m --- perform unsupervised hierarchical dynamic parsing and encoding (HDPE) for action recognition on the ChaLearn dataset

- selfclustering.m --- perform unsupervised parsing for a sequence by temporal clustering.

- computeWarpingPathtoTemplate_Eud_band_addc.m --- Compute the modified DTW alignment path from a sequence to a template


Dependency:

vlfeat-0.9.18
liblinear-1.93
libsvm-3.18



"ChaLearn_HDPE.m" is adapted from the following code:

- Chalearn_VideoDarwin.m

by Basura Fernando; this code can be downloaded from the website: https://bitbucket.org/bfernando/videodarwin

**Please check the licence in https://bitbucket.org/bfernando/videodarwin if you want to make use of this code.

############################################################################

4. Notice

1) The default parameters in this package are adjusted on the datasets used in the paper. You may need to adjust the parameters and the non-linear functions during the mean and rank pooling in the hierarchical encoding when applying it on other datasets. Please see the papers for details.

3) We adapted the code "Chalearn_VideoDarwin.m" provided by Basura Fernando which is publicly available. Please check the licence of it if you want to make use of this code or "ChaLearn_HDPE.m".

############################################################################

5. Citations

Please cite the following papers if you use the codes:

1) Bing Su, Jiahuan Zhou, Xiaoqing Ding, and Ying Wu, "Unsupervised Hierarchical Dynamic Parsing and Encoding for Action Recognition ", IEEE Trans. on Image Processing (TIP), 2017, 26(12), pp. 5784-5799. DOI: 10.1109/TIP.2017.2745212

2) Bing Su, Jiahuan Zhou, Xiaoqing Ding, Hao Wang, and Ying Wu, "Hierarchical Dynamic Parsing and Encoding for Action Recognition", Proc. European Conf. on Computer Vision (ECCV), 2016, pp. 202-217.


///////////////////////////////////////////////////
