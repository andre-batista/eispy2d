from chain import Train  
from eispy2d.api import api

params = {
    "wavelength": 1.0,
    "image_size": (4.0, 4.0),
    "observation_radius": 6.0,
    "background_permittivity": 1.0,
    "resolution": (30, 30),
    "noise_level": 1.0,    
    "disp": True,
    "contrast": 0.5,
    "object_size": 1.0,
    "number_sources": 10,
    "number_measurements": 10,
    
}




trainer = Train()

trainer.read("chain_data.pkl")

api.evaluate(trainer.solve, params)