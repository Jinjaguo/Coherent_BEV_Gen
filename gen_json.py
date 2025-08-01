import os
import json

image_dir = "../data/mini/samples/CAM_FRONT_"
output_file = "train.json"

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
image_files.sort()  

descriptions = [
    "An urban road passing through a block of red brick buildings, with a red light ahead, few pedestrians and vehicles, and cloudy weather.",
    "An urban intersection with red brick buildings, traffic lights showing red, and a few pedestrians on sidewalks under cloudy skies.",
    "A city street flanked by red brick buildings, with cars approaching the intersection and pedestrians crossing under cloudy weather.",
    "A red brick urban block with cars moving forward at a yellow light, scattered pedestrians, and an overcast sky.",
    "A city scene with a dark SUV entering the intersection, yellow traffic light ahead, and pedestrians walking on both sidewalks.",
    "An overcast street lined with red buildings, several moving vehicles, and people on sidewalks near a yellow light.",
    "A slightly brighter street scene with cars passing through the intersection, red brick buildings on both sides, and sparse foot traffic.",
    "A silver car approaching under a yellow traffic light in a red brick corridor, with pedestrians visible near building entrances.",
    "A cloudy day on a downtown street with red buildings and multiple cars in motion beneath a yellow light.",
    "A wide road flanked by tall red brick buildings with light traffic and a bus lane marking visible on the right side.",
    "A clearer view of an urban street stretching between red buildings, with cars in motion and distant skyscrapers visible ahead.",
    "A city street with red brick buildings, moderate traffic, and a black SUV turning left under cloudy weather.",
    "An urban road between red brick buildings, with a black SUV turning and several cars moving forward on a cloudy day.",
    "A downtown street flanked by red buildings, with parked cars on both sides and a black SUV turning at the intersection.",
    "An urban street with a black SUV turning left, other vehicles ahead, and overcast skies above red brick blocks.",
    "A red-brick-lined city road with a black SUV mid-turn, a few moving vehicles, and cloudy weather.",
    "A city scene with a black SUV crossing the road, silver cars ahead, and tall red buildings on both sides.",
    "An urban street with red brick walls, a black SUV completing a turn, and several parked cars under cloudy skies.",
    "A wide road lined with red buildings, light traffic, and a black SUV driving towards the center of the street.",
    "A downtown street with red brick blocks, a black SUV in the lane, and scattered vehicles under overcast weather.",
    "A city street stretching ahead between red brick buildings, moderate traffic, and a black SUV moving forward.",
    "A black SUV driving down a city street flanked by red brick buildings, with cloudy skies overhead.",
    "An urban road scene with the black SUV closer to the bottom of the image, cloudy weather and scattered cars ahead.",
    "A black SUV driving forward on an empty lane with a white car turning left ahead, red brick buildings on both sides.",
    "The black SUV continues driving in the middle of a city street, buildings ahead and overcast skies.",
    "A nearly straight-on view of the SUV driving in a central lane, some other vehicles seen further up the road.",
    "A slightly darker urban view, with the black SUV continuing straight, traffic lights and distant vehicles visible.",
    "City road continues with fewer nearby cars, black SUV slightly to the left lane, urban buildings line both sides.",
    "The SUV is farther away, heading down a road lined with old red buildings and cloudy weather.",
    "A distant view of the road with buildings and several cars ahead; the black SUV appears farther into the scene.",
    "A SUV fades into background traffic; buildings and a white van ahead as the road continues straight.",
    "A city street with a black SUV driving away, tall red brick buildings on both sides, cloudy weather.",
    "A urban road scene with the SUV farther ahead, more cars visible in distance, and cloudy sky above red brick surroundings.",
    "A long road ahead with several vehicles in view, red buildings line the path, and the SUV is barely visible.",
    "A SUV is on the city road; urban buildings and cars appear in the middle ground under overcast skies.",
    "A road straight into the horizon, with vehicles ahead and the cityscape becoming denser in the distance.",
    "A broader view of the street, traffic increasing in the distance, still within a red-brick-lined corridor.",
    "An approaching intersection with dense red and brown buildings ahead; traffic is visible under gray skies.",
    "Cars are seen ahead near a larger building structure; more distant skyscrapers begin to appear on the horizon.",
    "A urban environment becomes denser, with more modern structures ahead and traffic continuing under a cloudy sky.",
    "A farther view into the city, with a prominent tall white building visible at the end of the street; traffic and urban density increase.", 
    "A straight road with light grey office buildings on both sides on a sunny day, with cars parked on the roadside and sparse traffic ahead.",
    "A sunny commercial street with rows of light grey buildings on the left and a white high-rise in the distance in the middle.",
    "A straight asphalt road runs through the grey office building area, with few vehicles on the road and a clear sky.",
    "Industrial style street scene, neatly parked cars on the left, low-rise factory buildings on the right, and scattered vehicles under the clear sky.",
    "A wide city road with glass curtain wall buildings on the left and clear outlines of high-rise buildings in the distance.",
    "A commercial road on a sunny day, with a solid yellow line in the center extending into the distance and tall white buildings as the background.",
    "A straight road with modern office buildings on both sides, densely parked vehicles on the roadside and smooth traffic.",
    "A street scene on a bright weekday, with a line of cars on the left, a sidewalk and small trees on the right, and a high-rise building standing in the distance.",
    "A main city road under the sun, with few vehicles on the road, simple building facades, and clear sightlines.",
    "A road in a modern industrial park, with concrete buildings and parked trucks on the left, parking lots and green plants on the right, and a clear sky with few clouds."
]


data = []
for img, desc in zip(image_files, descriptions):
    item = {
        "image": os.path.join(image_dir, img),
        "prompt": "Use a complete sentence to describe the scene in the picture:",
        "answer": desc
    }
    data.append(item)

# 保存
with open(output_file, "w") as f:
    for item in data:
        json.dump(item, f)
        f.write('\n')
