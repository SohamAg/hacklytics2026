# Google Lens Feature Setup Guide

## Overview
Your heart visualization now has a Google Lens-style feature that allows you to:
- **Select areas** of the 3D heart model
- **Analyze** selected parts with AI-powered insights
- **Get information** about cardiac anatomy using RAG (Retrieval-Augmented Generation)

## How to Use

### 1. Activate Lens Mode
Click the **🔍 Lens** button in the toolbar (top-left of the canvas). The button will show **✓ Lens** when active.

### 2. Select an Area
- **Click and drag** on the canvas to draw a selection box around the heart parts you want to analyze
- A blue selection box will appear as you drag
- Release to analyze the selected region

### 3. View AI Analysis
A popup will appear with information about the selected part(s):
- Part names and geometry info
- Cardiac anatomy details (if Gemini API is connected)
- Clinical significance (with full API key)

## Setting Up Gemini API (Optional)

### For Full AI Features:
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key (free tier available)
3. When you first use the Lens feature, you'll be prompted to enter your API key
4. The key will be saved in your browser's localStorage

### Alternative: Use Environment Variable
If you're using a build tool (Vite/Next.js), create a `.env.local` file:
```
VITE_GEMINI_API_KEY=your_api_key_here
```

## RAG Data Integration

The system automatically loads and uses data from:
- `data-processing/heart_features.csv` - Cardiac feature data
- `data-processing/hvsmr_clinical.csv` - Clinical metrics
- `data-processing/hvsmr_technical.csv` - Technical specifications

This data is used as context when Gemini generates insights about selected parts.

## Architecture

### Files Added/Modified:

**New File: `lens-module.js`**
- Handles all lens functionality (no dependencies on heart-scene.js)
- Canvas selection visualization
- Raycasting to identify selected meshes
- Gemini API integration
- RAG context building
- Popup UI management

**Modified: `index.html`**
- Added lens button to toolbar
- Added popup styles
- Added lens module initialization script
- Minimal structural changes

**Modified: `heart-scene.js`**
- Single line added: lens module initialization after model loads
- No impact on existing functionality

### How It Works

1. **Selection Detection:**
   - Overlay canvas captures mouse drag
   - Uses raycasting to identify Three.js meshes in selection area
   - Collects geometry and material info

2. **RAG Processing:**
   - CSV files parsed into structured data
   - Data summaries included in API context
   - Provides clinical background to AI

3. **AI Analysis:**
   - Gemini API generates insights based on selection
   - Falls back to basic anatomy knowledge if API unavailable
   - Results formatted and displayed in popup

4. **No Server Required:**
   - All processing happens client-side
   - Direct API calls to Google Generative AI
   - Data stored in browser localStorage

## Troubleshooting

### Lens button not working?
- Check browser console for errors
- Ensure JavaScript is enabled
- Try clearing browser cache

### AI analysis showing generic info?
- You haven't set a Gemini API key yet
- Click the Lens button again to provide your key
- Without an API key, you'll see basic anatomical information instead

### Selection not detecting parts?
- Make sure you're dragging a reasonable size box
- Parts need to be under your selection area
- Try dragging in the center-bottom (where apex is)

### CSV RAG data not loading?
- Check that CSV files exist in `data-processing/`
- Check browser console for CORS warnings
- Data loading is optional; feature still works without it

## Security Notes

- Your Gemini API key is stored in browser localStorage
- Keep your API key private (never commit to git)
- Consider using environment variables in production
- API key is only sent to Google's servers

## Future Enhancements

Potential improvements:
- Custom RAG documents (upload medical papers)
- Speech input for voice queries
- Multi-language analysis
- Comparison mode (compare two selections)
- Heatmap visualization of selection importance
