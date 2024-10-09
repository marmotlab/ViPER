# ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning

This repository hosts the code for [**ViPER**](https://openreview.net/pdf?id=EPujQZWemk), accepted for [CoRL 2024](https://www.corl.org/).

<div>
   <img src="utils/media/demo.gif" height="250"/>
   <img src="utils/media/demo_large.gif" height="250"/>
</div>

ViPER is a neural framework for visibility-based pursuit-evasion, where agents cooperatively search for worst-case evaders.
A team of agents methodically explores and clears the entire environment by expanding frontiers, turning all contaminated areas (white) into cleared ones (green).

## Setup instructions

Use conda and pip to setup environments:

```bash
conda create -n viper python=3.11 scikit-image imageio tensorboard matplotlib pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate viper
pip install ray opencv-python wandb 
```

## Evaluation

### Download pretrained model and dataset

```bash
bash ./utils/download.sh
```

Set appropriate parameters in `test_parameter.py` and run `test_driver.py` to evaluate.

### Interactive demo

You can also create your own map by running `viper_demo.py`, which opens a canvas for you to draw on.

- Use **Obstacle** and **Free Space** brushes to draw your map. Adjust the brush size with the thickness slider.
- Click **Reset** to clear the canvas, setting it entirely to obstacles or free space.
- Click **Place Agents** to place multiple agents in the free space.
- Click **Play** to observe how ViPER agents plan their paths. The canvas will close and the interactive demo will play automatically.

Alternatively, you can save the map that you created.

- Click **Start Position** to place the starting position of agents.
- Click **Save Map** before closing the canvas. Your map will be saved as `maps_spec/map.png`. 

## Training

Make sure you have downloaded the map dataset.
Set appropriate parameters in `parameter.py` and run `driver.py` to train the model.


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{wang2024viper,
  title={ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning},
  author={Wang, Yizhuo and Cao, Yuhong and Chiun, Jimmy and Koley, Subhadeep and Pham, Mandy and Sartoretti, Guillaume},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024}
}
```

Authors:
[Yizhuo Wang](https://www.yizhuo-wang.com/),
[Yuhong Cao](https://www.yuhongcao.online/),
[Jimmy Chiun](https://www.linkedin.com/in/jimmychiun/),
[Subhadeep Koley](https://www.linkedin.com/in/subhadeep-koley-70251b1bb/),
[Mandy Pham](https://www.linkedin.com/in/phamandy24/),
[Guillaume Sartoretti](https://cde.nus.edu.sg/me/staff/sartoretti-guillaume-a/)
