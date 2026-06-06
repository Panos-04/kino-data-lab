# KINO Data Lab

A full-stack data analysis project for studying KINO draw trends using Django, Django REST Framework, and React.

## Features

- Imports historical KINO draw data from the public OPAP/Allwyn API
- Stores raw draw results in a Django database
- Builds sliding-window heatmap analysis
- Supports 20-game windows moving by 10 games
- Supports 10-game windows moving by 5 games
- Displays KINO numbers as visual trend frames
- Heat scale from 0 to 11+
- Clickable balls show split data for the selected frame

## Stack

- Django
- Django REST Framework
- React
- TypeScript
- Vite
- SQLite for local development

## Current Status

Stable development checkpoint.

Implemented:
- draw importer
- range importer
- frequency analyzer
- window builder
- REST API for window data
- React trend-frame board
- split view on clicked numbers

Next planned:
- relation analysis between selected high/mid/low heat numbers
- saved relation tables
- click-based relation panel
- larger-scale backtesting