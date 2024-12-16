from app import create_app

try:
    app = create_app()
except Exception as e:
    print(f"Error creating the Flask app: {e}")

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error running the Flask app: {e}")
