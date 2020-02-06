# ObjectRemoval

Based on code from: \
https://github.com/StanfordVL/GibsonEnv \
https://github.com/StanfordVL/3DSceneGraph \
https://github.com/seoungwugoh/opn-demo \
The licenses are included in the above links.

## Installation

Follow the instructions in the 3 internal README files to download and install the relevant data, code, models and dependencies. \
Save the 3DSceneGraph folder ("verified_graph") under GibsonEnv/gibson/assets.

## Usage

Build the panoramic for all models masks with:

```bash
cd 3DSceneGraph 
bash load.sh
```
or for one model with:

```bash
cd 3DSceneGraph 
python load.py --model model_name
```



To run object removal: \
update the 'model_id' field in opn/play_drone_camera.yaml with the model name.

Run:
```bash
cd opn 
python gibson_inpaint.py
```
