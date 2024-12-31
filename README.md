# **FPL Team Viewer**

A web application that dynamically fetches and displays your Fantasy Premier League (FPL) team lineup and bench for the current gameweek, with a visually appealing interface and player information modal.

## **Features**

- **Dynamic Lineup Display**: Displays the starting lineup and bench for the current gameweek.
- **User ID Input**: Update the displayed team by entering a new FPL User ID.
- **Player Info Modal**: Click on a player to view detailed stats, including price, form, points, and status.
- **Color-Coded Status**: Players with injuries or doubts are highlighted with red or yellow backgrounds.
- **Responsive Design**: Works on both desktop and mobile browsers.
- **Expandable**: Modular structure for easy feature additions or customizations.


## **Getting Started**

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nishantgerald/fpl.git
   cd fpl
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Application**
   - Ensure you have an active internet connection for the app to fetch data from the Fantasy Premier League API.

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Open in Browser**
   - Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

## **Usage**

1. **View Your Team**
   - Enter your FPL User ID (default is currently my FPL user ID `3022850`) in the input box at the top left. (feel free to change the default)
   - Click **"Update Team"** to view the current gameweek's lineup and bench.

2. **Explore Player Details**
   - Click the **info (i)** button on any player to open a modal with detailed stats.

3. **Status Colors**
   - **Red**: Injured, Suspended, or Unavailable players.
   - **Yellow**: Doubtful players.
   - **White**: Available players.

## **Code Customization**

### 1. **Add New Tabs**
- Modify `formation_with_bench.html` to include new tab buttons.
- Add logic to `app.py` to support additional views or features.

### 2. **Change Styles**
- Update `styles.css` to customize colors, layout, or fonts.

### 3. **Improve API Handling**
- Refactor `fetch_player_data()` and related functions in `app.py` to handle API rate limits or failures.

### 4. **Add New Player Info**
- Extend `/player-stats/<player_id>` in `app.py` to fetch and display additional player statistics.


## **Known Issues**

1. **API Dependency**: 
   - If the Fantasy Premier League API is down, the app won't work.
   - Solution: Implement caching or a fallback mechanism.

2. **Mobile Responsiveness**:
   - Some UI elements may require further adjustment for smaller screens.

3. **Player Status Handling**:
   - If new status codes are introduced, update `STATUS_MAP` in `app.py`.


## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.


## **Contact**

For any issues or feature requests, feel free to create an issue in this repository.