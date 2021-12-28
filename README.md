# Audio Super-Resolution
A project concerning the super-resolution of audio signals i.e. increasing the sample rate of an audio signal by using a U-Net architecture and eventually a generative adversarial network like SRGAN (only as an experiment, considering that SRGAN hasn't been designed for this task).


## Dataset
The dataset used will be VCTK, but if there's enough time to research more about super-resolution applied on music, the dataset could be either MagnaTagATune or it could be generated with the Python module [Spotipy](https://spotipy.readthedocs.io/en/2.19.0/) which works with the Spotify API.


## Converting a spectrogram to audio
The Python module [librosa](https://librosa.org/doc/latest/index.html) can transform an audio signal into a spectrogram and vice-versa
by using the [Griffin-Lim algorithm](https://stackoverflow.com/questions/61132574/can-i-convert-spectrograms-generated-with-librosa-back-to-audio).


## Main references
 - [Audio Super-Resolution Using Neural Nets](https://arxiv.org/pdf/1708.00853v1.pdf)

## Related papers that could be used as references

### Audio signals
 - [Phase-aware music super-resolution using generative adversarial networks](https://arxiv.org/pdf/2010.04506.pdf)
 - [Time-frequency Networks For Audio Super-Resolution](https://teckyianlim.me/audio-sr/res/3828.pdf)
 - [MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation](https://arxiv.org/pdf/1703.10847.pdf)
 - [Adversarial Audio Super-Resolution With Unsupervised Feature Losses](https://openreview.net/pdf?id=H1eH4n09KX)
 - [On the evaluation of generative models in music](https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2018/11/postprint.pdf)
 - [INCO-GAN: Variable-Length Music Generation Method Based
on Inception Model-Based Conditional GAN](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj62YWr14_0AhXch_0HHf0dDckQFnoECAQQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2227-7390%2F9%2F4%2F387%2Fpdf&usg=AOvVaw1Bt1-i7aM7Fp8OmwRJ2GtX)
 - [WSRGlow: A Glow-based Waveform Generative Model for Audio Super-Resolution](https://arxiv.org/abs/2106.08507)
 - [Self-Attention for Audio Super-Resolution](https://arxiv.org/pdf/2108.11637v1.pdf)
 - [On Filter Generalization for Music Bandwidth Extension Using Deep Neural Networks](https://arxiv.org/pdf/2011.07274v2.pdf)
 - [NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling](https://arxiv.org/pdf/2104.02321v2.pdf)
 - [Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulation](https://arxiv.org/pdf/1909.06628v3.pdf)
 - [Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution](https://arxiv.org/pdf/2111.00195.pdf)
 - [An investigation of pre-upsampling generative modelling and Generative Adversarial Networks in audio super resolution](https://arxiv.org/pdf/2109.14994.pdf)
 - [Adversarial Training for Speech Super-Resolution](https://www.researchgate.net/publication/332201260_Adversarial_Training_for_Speech_Super-Resolution)
 - [Super-Resolution for Music Signals Using Generative Adversarial Networks](https://www.researchgate.net/publication/354040914_Super-Resolution_for_Music_Signals_Using_Generative_Adversarial_Networks)
 - [Bandwidth extension on raw audio via generative adversarial networks](https://arxiv.org/pdf/1903.09027.pdf)


### Image super-resolution 
 - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802.pdf)

### Time-series
 - [Imaging Time-Series to Improve Classification and Imputation](https://arxiv.org/pdf/1506.00327.pdf)

### Generative models
 - [Wasserstein GANs](https://arxiv.org/pdf/1701.07875.pdf)
 - [STGAN: A Unified Selective Transfer Network
for Arbitrary Image Attribute Editing](https://arxiv.org/pdf/1904.09709.pdf)

### Possibly useful
 - [Audio Super-Resolution Using Analysis Dictionary
Learning](http://personal.ee.surrey.ac.uk/Personal/W.Wang/papers/DongWC_DSP_2015.pdf)

### Articles
 - [Using Deep-Learning to Reconstruct High-Resolution Audio](https://blog.insightdatascience.com/using-deep-learning-to-reconstruct-high-resolution-audio-29deee8b7ccd)

## Possible uses
 - increasing sample rate of music
 - voice-over-IP
 - improved speech recognition
 - remastering audio from old movies

## To do:
 - [X] Find and read 2-3 online tutorials or research papers that solve the same problem. 

 - [X] Find a way to gather data

 - [X] Write the introduction section 

 - [X] Write the data generator scripts (obtaining the low-res/high-res pairs of audio clips)

 - [X] Write the training/testing scripts
 
