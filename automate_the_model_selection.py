def find_best_model(models):
    # Third part - Non-Mandatory Assignments
    # Automate the model selection procedure

    max = 0
    for i in range(len(models)):
        if models[i][2] > models[max][2]:
            max = i

    print("Best model found:")
    print(models[max][0])
    return models[max][1]