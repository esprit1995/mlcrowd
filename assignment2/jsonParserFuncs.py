import json
from pathlib import Path


def get_coordinates(path):
    """
    get initial pedestrian coordinates from the file on the specified path
    :param path: path to the trajectories file. With filename included.
    :return: arrays of float coordinates xs and ys.
    """

    print('******Getting trajectories from ' + str(path))
    with open(str(Path(path)), "r") as f:
        line = f.readline()
        print(line)
        timeStep = "1"

        xs = list()
        ys = list()
        while line and timeStep == "1":
            line = f.readline()
            if line:
                timeStep, x, y = line.split(" ")[0], line.split(" ")[2], line.split(" ")[3]
                if timeStep == "1":
                    xs.append(x)
                    ys.append(y)
        f.close()
        print('*****Done.')
    return xs, ys


def add_pedestrian(scenario_json,
                   x, y,
                   xvelocity=1,
                   yvelocity=1,
                   targets=[],
                   minimumspeed=1,
                   maximumspeed=1,
                   acceleration=0,
                   freeflowSpeed=1):
    """
    Adds a pedestrian in the scenario_json with specified parameters
    Accepts parameters to set and a scenario in json format
    Returns updated dynamicElements attribute
    """

    dynamicElements = scenario_json["scenario"]["topography"]["dynamicElements"]
    basePedestrian = {
        "source": None,
        "targetIds": [],
        "position": {
            "x": 0.0,
            "y": 0.0
        },
        "velocity": {
            "x": 0.0,
            "y": 0.0
        },
        "nextTargetListIndex": 0,
        "freeFlowSpeed": 1.0,
        "attributes": {
            "id": -1,
            "radius": 0.2,
            "densityDependentSpeed": False,
            "speedDistributionMean": 1.0,
            "speedDistributionStandardDeviation": 0.0,
            "minimumSpeed": 1.0,
            "maximumSpeed": 1.0,
            "acceleration": 0.0,
            "footStepsToStore": 4,
            "searchRadius": 1.0,
            "angleCalculationType": "USE_CENTER",
            "targetOrientationAngleThreshold": 45.0
        },
        "idAsTarget": -1,
        "isChild": False,
        "isLikelyInjured": False,
        "mostImportantEvent": None,
        "salientBehavior": "TARGET_ORIENTED",
        "groupIds": [],
        "trajectory": {
            "footSteps": []
        },
        "groupSizes": [],
        "modelPedestrianMap": None,
        "type": "PEDESTRIAN"
    }

    # set location
    basePedestrian["position"]["x"] = x
    basePedestrian["position"]["y"] = y

    # set targets
    basePedestrian["targetIds"] = [i for i in targets]

    # set velocity
    basePedestrian["velocity"]["x"] = xvelocity
    basePedestrian["velocity"]["y"] = yvelocity

    # set speed and acceleration
    basePedestrian["attributes"]["minimumSpeed"] = minimumspeed
    basePedestrian["attributes"]["maximumSpeed"] = maximumspeed
    basePedestrian["attributes"]["acceleration"] = acceleration

    # set freeFlowSpeed
    basePedestrian["freeFlowSpeed"] = freeflowSpeed

    # add the resulting pedestrian
    dynamicElements.append(basePedestrian)

    return dynamicElements


def replace_source_with_pedestrians(scenarioname,
                                    trajectoryname,
                                    pathscenario='scenarios',
                                    pathtraj='trajectories',
                                    newname="task5Updated.scenario",
                                    outputdir="outputs",
                                    targets=[1]):
    """
    For scenario located at 'pathfilename', delete sources and insert pedestrians in the coordinates recorded in pathfilenametrakj
    :param scenarioname: filename of the scenario we want to change. String
    :param trajectoryname: filename of the trajectories we want to use. String.
    :param pathscenario: path to the scenario we want to change. String
    :param pathtraj: path to the trajectories. String
    :param outputdir: where to put the new scenario file. String
    :param newname: how to name the new file. Has a default.
    :param targets: targets of the pedestrians. Array of ints. Has a default.
    :return: void. Alters the file.
    """

    newpath = Path(pathscenario) / scenarioname
    print(newpath)
    with open(str(newpath), "r") as f:
        datastore = json.load(f)
    f.close()

    # update name.
    datastore["name"] = newname

    # destroy the sources
    datastore["scenario"]["topography"]["sources"] = []

    # add pedestrians
    xs, ys = get_coordinates(Path(pathtraj) / trajectoryname)
    for i in range(len(xs)):
        datastore["scenario"]["topography"]["dynamicElements"] = add_pedestrian(datastore, x=xs[i], y=ys[i],
                                                                                targets=targets)
    with open(str(Path(outputdir) / newname), "w") as f:
        json.dump(datastore, f)
        f.close()
