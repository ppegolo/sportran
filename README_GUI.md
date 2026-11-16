# *SporTran* GUI


## Installation
See `README.md`.

## Usage
1. Run `sportran-gui` from the command line.
2. Select the input file with the "`...`" button. After you select the file a preview will be shown if possible. If a binary file is selected the preview will not update, and a message will pop up.
3. Select the input format:
   - "`table`" is the standard column-formatted data file, where the first line contains the names of each column. In order to be recognized, vector data (e.g. the 3 Cartesian components of a current) must be named as "`name_[1] name_[2] name_[3]`".
   - "`dict`" is a NumPy dictionary (`numpy.save()`) containing binary data. This type of file can be generated when you press "export data" in the "File" menu. A binary file is faster to load.
4. Press "`Next`". If the selected file type is wrong, an error message pops up and the program returns to the previous step.
5. Read the instructions and select the type of data corresponding to each column.
6. Select the units.
7. Press "`Next`".
8. Input the required values (if not already filled with the information retrieved from the input file).
9. "`Filter width`" can be left as is. It is used for visualization and can be changed later.
10. Select the "`Resampling frequency`" with the cursor. You can press "`Resample`" to see the effect. This is necessary to avoid running the analysis on useless portions of the spectrum. You can select the first important feature of the plot near zero frequency (see manual and docs for reference).
11. Press "`Next`".
12. Inspect the result. It is possible to tweak the default number of cepstral coefficients and visualize the effect.

At any time it is possible to save the current status by pressing "`Export data`" from the menu. All the parameters can be automatically reloaded if you select the generated file as input.
