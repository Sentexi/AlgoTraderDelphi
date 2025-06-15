import xgboost as xgb

# Path to the saved model file
model_path = "barebones.model"

try:
    # Load the model
    model = xgb.Booster()
    model.load_model(model_path)

    # Display general model information
    print("Model loaded successfully!")

    # Extract model parameters
    params = model.attributes()
    print("\nModel Attributes:")
    for key, value in params.items():
        print(f"{key}: {value}")

    # Get the number of trees
    num_boosted_rounds = len(model.get_dump())
    print(f"\nNumber of Boosted Rounds (Trees): {num_boosted_rounds}")

    # Display feature importance (if available)
    importance = model.get_score(importance_type='weight')
    print("\nFeature Importance (Weight):")
    for feature, score in importance.items():
        print(f"{feature}: {score}")

    # Display the model dump (optional, large output)
    print("\nModel Dump (First 5 Trees):")
    model_dump = model.get_dump()
    for i, tree in enumerate(model_dump[:5]):
        print(f"Tree {i+1}:\n{tree}\n")

except Exception as e:
    print(f"Failed to load the model: {e}")
