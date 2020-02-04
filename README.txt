OVERVIEW:

TabOut uses Python, TensorFlow, and Keras to automate guitar tablature from audio files 
9k training images + 3k test samples from GuitarSet (https://zenodo.org/record/1422265#.XjjU0xNKjVq) indicate average accuracy of 81% 
I got a lot of help from this: http://nemisig2019.nemisig.org/images/kimSlides.pdf


NITTY GRITTY:

The GuitarSet data contains all the notes as MIDI values as well as the duration and relative time of each note.
These files are transformed into images with the librosa library available at librosa.github.io/librosa. 
The constant-Q transform is applied to the images to train the CNN. 
The Fret variable contains a 6x18 matrix of MIDI values representing the 6 strings and 18 frets of a standard guitar. 
The program then determines all possible locations of the unique notes, and then the most likely combination of frets and strings. 
The Keras functional API is used to create multitask classification model, each string is a task 


ISSUES: 

Need to make better user interface, current implementation is optimized for personal use/testing/debugging (e.g. filepath is hardcoded)
having trouble accounting for the duration the note, necessary for good full length tabs 
specific voicings, i.e. right chord but wrong shape. not a huge deal but might be nice