# Counter Factual Toolbox

This project consists of two parts, first the toolbox for generating counter factual images that 
improve the model prediction and second the code for using that toolbox on a focal cortical dysplasia dataset.


## Counter Factual Toolbox
The Counter Factual Toolbox provides a number of methods to generate a so called "counter factual" image.
This is an image that is changed in a way that the models prediction is very close the target. 
The Counter Factual Toolbox provides three ways to generate this image. By smoothly adopting to every 
image pixel, by using a deformation and by using an adversarial that controls the image to stay in the image domain.

An example on how to use the toolbox can be found in the focal cortical dysplasia Use-Case.

### Install
The Counter Factual Toolbox is also installable via pip:
```` bash
git clone https://github.com/NiklasKerkfeld/SegmentationBacktrackingSandbox.git && cd SegmentationBacktrackingSandbox
pip install .
````

### Generate

To generate a counter factual image the Generator classes can be used:
```` python
    generator = SmoothChangeGenerator(model, image, target)
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)
    generator.generate(optimizer, steps=100)
```` 
The `log_and_visualize` function of the generator visualizes the results.

## Focal cortical dysplasia Use-Case
I used the 
["An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II"](https://openneuro.org/datasets/ds004199/versions/1.0.6)
and the ["nnUNet_fcd"](https://gitlab.com/lab_tni/projects/nnunet_fcd) UNet model as a Use-Case for my model. 

The `FCD_Usecase/scripts/generate.py` shows how to generate counter factual images for this dataset using different methods.
      
![Example](assets/overview.png)

            
        

          
    


    
