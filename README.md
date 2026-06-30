# Track Forge

Upload a GPX/FIT file or connect to Garmin Connect to turn an activity into shareable social assets.

## Features

- Garmin Connect login plus GPX/FIT upload
- Latest 100 Garmin activities
- Transparent social-ready stills and animations
- Optional black-and-white map background
- Toggleable and reorderable stats
- GIF, WebM, MP4, and PNG export buttons

## Run

```bash
streamlit run streamlit_app.py
```

## Secrets

Generate a local secrets file with defaults and examples:

```bash
python scripts/generate_secrets.py
```

Or copy the template directly:

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

The app expects `EMAIL` and `PASSWORD` under the `[garmin]` section.

## Notes

- Transparency is preserved best in PNG, GIF, and WebM.
- MP4 export is included for compatibility and may be flattened by some players.
- Put your Garmin Connect credentials in `.streamlit/secrets.toml`.
