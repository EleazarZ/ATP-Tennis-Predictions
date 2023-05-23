#! /usr/bin/env python
"""Python main to run the flask app"""
from predict_api import app

if __name__ == "__main__":
    app.run(debug=True)
