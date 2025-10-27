# Project Structure

## Main Program - timer.py

The main entry point for the experiment program. run this file to run the experiment.

## Plotting - view_csv.py
This file allows for data inspection by plotting selected channels from recorded data files

## Configuration - experiment_settings.py
Change the constants in this file configure the UI and file pathing for the experiment

## Practice Program - practice_timer.py

Run this file to show the practice experiment UI, which shows movement GIFS.
## Core Package

Contains all helpers related to experiment flow and the graphic interface.

## Util Package
Contains all helpers for background tasks, such as device communication, recording management, and file saving.

## Movement Library
Contains the movement images used for the experiment

## Dependencies - requirements.txt
If any new dependencies are added, they should be defined here to ensure future dependency installations are complete

## Docs 

The docs package contains the markdown for the pages on this site. mkdocs.yml and requirements-docs.txt define the dependencies and behaviour of the docs, but they should not need to be changed
