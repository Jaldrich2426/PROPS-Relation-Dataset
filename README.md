# PROPS Relation Dataset
Props Dataset formatted to work with SORNet, a working example of which can be found at our fork [here](https://github.com/Jaldrich2426/sornet).

## Installation

In this repo, run:
```bash
pip install -e .
```

It is recomended to have already setup SORNet's conda package as they share many dependencies

## Dataset Download

To download the objects formatted for PROPS, download this file and extract the objects directory to the workspace using this pip package

https://drive.google.com/file/d/1rEn88wnhNdN8BsghrHNyaOdXeCRIi8F3/view?usp=sharing

## PROPS Usage

An example usage of the Props Relation dataset is provided in its source file, as well as three examples in our sornet fork:

[training](https://github.com/Jaldrich2426/sornet/blob/main/train_props.py)

[testing](https://github.com/Jaldrich2426/sornet/blob/main/test_props.py)

[visualizing](https://github.com/Jaldrich2426/sornet/blob/main/visualize_props.py)

## Custom Usage

To use this framework with another dataset, simply create a new file and overload the BaseRelationDataset class, following the exmaple in PropsRelationDataset. You need to overload each method that raises a "NotImplemented" error in the same manner as which PROPS does. If there is an existing dataset manager class, initialize it in 

```_init_parent_dataset()```

to make the implementation easier. Otherwise, load the appropriate file information in each class method and the base class should handle the relation information automatically, provided that the camera frame uses the standard notation.

If you would like to add further relations, simply overload the ```get_spatial_relations()``` method, and watch for any locations a "4" was hardcoded for the number of locations.