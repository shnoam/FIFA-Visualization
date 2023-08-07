import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.express as px


fifa17 = pd.read_csv('fifa17.csv')
fifa18 = pd.read_csv('fifa18.csv')
fifa19 = pd.read_csv('fifa19.csv')
fifa20 = pd.read_csv('fifa20.csv')
fifa21 = pd.read_csv('fifa21.csv')
fifa22 = pd.read_csv('fifa22.csv')

# FIFA 17
fifa17_defender = pd.read_csv('fifa17_defender.csv')
fifa17_midfielder = pd.read_csv('fifa17_midfielder.csv')
fifa17_forward = pd.read_csv('fifa17_forward.csv')
fifa17_goalkeeper = pd.read_csv('fifa17_goalkeeper.csv')

# FIFA 18
fifa18_defender = pd.read_csv('fifa18_defender.csv')
fifa18_midfielder = pd.read_csv('fifa18_midfielder.csv')
fifa18_forward = pd.read_csv('fifa18_forward.csv')
fifa18_goalkeeper = pd.read_csv('fifa18_goalkeeper.csv')

# FIFA 19
fifa19_defender = pd.read_csv('fifa19_defender.csv')
fifa19_midfielder = pd.read_csv('fifa19_midfielder.csv')
fifa19_forward = pd.read_csv('fifa19_forward.csv')
fifa19_goalkeeper = pd.read_csv('fifa19_goalkeeper.csv')

# FIFA 20
fifa20_defender = pd.read_csv('fifa20_defender.csv')
fifa20_midfielder = pd.read_csv('fifa20_midfielder.csv')
fifa20_forward = pd.read_csv('fifa20_forward.csv')
fifa20_goalkeeper = pd.read_csv('fifa20_goalkeeper.csv')

# FIFA 21
fifa21_defender = pd.read_csv('fifa21_defender.csv')
fifa21_midfielder = pd.read_csv('fifa21_midfielder.csv')
fifa21_forward = pd.read_csv('fifa21_forward.csv')
fifa21_goalkeeper = pd.read_csv('fifa21_goalkeeper.csv')

# FIFA 22
fifa22_defender = pd.read_csv('fifa22_defender.csv')
fifa22_midfielder = pd.read_csv('fifa22_midfielder.csv')
fifa22_forward = pd.read_csv('fifa22_forward.csv')
fifa22_goalkeeper = pd.read_csv('fifa22_goalkeeper.csv')

goalkeeper_lst = [(fifa17_goalkeeper, "fifa 17 goalkeeper"), (fifa18_goalkeeper, "fifa 18 goalkeeper"), (fifa19_goalkeeper, "fifa 19 goalkeeper"), (fifa20_goalkeeper, "fifa 20 goalkeeper"), (fifa21_goalkeeper, "fifa 21 goalkeeper"), (fifa22_goalkeeper, "fifa 22 goalkeeper")]
defender_lst = [(fifa17_defender, "fifa 17 defender"), (fifa18_defender, "fifa 18 defender"), (fifa19_defender, "fifa 19 defender"), (fifa20_defender, "fifa 20 defender"), (fifa21_defender, "fifa 21 defender"), (fifa22_defender, "fifa 22 defender")]
midfielder_lst = [(fifa17_midfielder, "fifa 17 midfielder"), (fifa18_midfielder, "fifa 18 midfielder"), (fifa19_midfielder, "fifa 19 midfielder"), (fifa20_midfielder, "fifa 20 midfielder"), (fifa21_midfielder, "fifa 21 midfielder"), (fifa22_midfielder, "fifa 22 midfielder")]
forward_lst = [(fifa17_forward, "fifa 17 forward"), (fifa18_forward, "fifa 18 forward"), (fifa19_forward, "fifa 19 forward"), (fifa20_forward, "fifa 20 forward"), (fifa21_forward, "fifa 21 forward"), (fifa22_forward, "fifa 22 forward")]

# Set the title of the app
st.title('FIFA Visualization')

# Option selection
option = st.selectbox('Select a graph to display', options=[
    "Abilities impact on Overall rating",
    "Correlation between Personal Abilities and Overall Rating",
    "The Dream Team",
    "Player's potential",
    "Nationality Distribution among Top Players",
    "Who is better in FIFA, Messi or Ronaldo?",
])

# Option 1: Relationship between individual ability ratings and overall rating
if option == "Abilities impact on Overall rating":
    def plot_summary_radar_chart(lst, main_title):
        all_ability_ratings = []
        all_overall_ratings = []

        # Iterate over the datasets
        for dataset, title in lst:
            # Select personal ability ratings and overall ratings for each player
            ability_ratings = dataset[
                ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
                 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                 'GKKicking', 'GKPositioning', 'GKReflexes']]
            overall_ratings = dataset['Overall']

            # Append the ability ratings and overall ratings to the lists
            all_ability_ratings.append(ability_ratings)
            all_overall_ratings.append(overall_ratings)

        # Concatenate all the ability ratings and overall ratings
        all_ability_ratings = pd.concat(all_ability_ratings)
        all_overall_ratings = pd.concat(all_overall_ratings)

        # Calculate the average ratings for each ability
        average_ratings = all_ability_ratings.mean()
        average_overall = all_overall_ratings.mean()

        # Normalize the average ratings and overall rating
        normalized_ratings = average_ratings / average_overall

        # Set up the radar chart
        categories = normalized_ratings.index
        values = normalized_ratings.values

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])
        ax.set_title(main_title)

        plt.show()


    # Define the function for plotting the correlation charts
    st.subheader("relationship between the player's individual ability ratings and his overall rating in the game")
    st.markdown("The graph below shows the radar chart which represents the impact of each feature on the Overall rating.\n"
                "The more the feature pulls the radar image, the stronger the impact on the Overall rating.")

    # Create the position selection dropdown
    position = st.selectbox("Select Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"])

    # Assign the appropriate list based on the selected position
    if position == "Goalkeeper":
        position_lst = goalkeeper_lst
    elif position == "Defender":
        position_lst = defender_lst
    elif position == "Midfielder":
        position_lst = midfielder_lst
    elif position == "Forward":
        position_lst = forward_lst

    # Plot the radar chart for the selected position
    plot_summary_radar_chart(position_lst, f"{position} : Impact of Abilities on Overall Rating - Average of All Years")

    # Display the radar chart using st.pyplot()
    st.pyplot(plt)


# Option 2: Correlation between personal abilities and overall rating
elif option == "Correlation between Personal Abilities and Overall Rating":
    def plot_correlation(lst, position):
        fig, axes = plt.subplots(3, 2, figsize=(12, 16))
        axes = axes.flatten()

        for i, (dataset, title) in enumerate(lst):
            ability_ratings = dataset[
                ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
                 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                 'GKKicking', 'GKPositioning', 'GKReflexes']]
            overall_ratings = dataset['Overall']

            # Calculate correlation between personal abilities and overall rating
            correlation_matrix = ability_ratings.corrwith(overall_ratings)

            # Sort the correlations in descending order
            sorted_correlations = correlation_matrix.abs().sort_values(ascending=False)

            # Plot the sorted correlations with reversed colors
            ax = axes[i]
            sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, color='steelblue', ax=ax)
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Personal Ability')
            ax.set_title(title)

            # Add numbers on the bars
            for j, correlation in enumerate(sorted_correlations.values):
                ax.text(correlation, j, f'{correlation:.2f}', ha='left', va='center')

        plt.suptitle(f'{position}: Correlation between Personal Abilities and Overall Rating\n', fontsize=16,
                     fontweight='bold')
        plt.tight_layout()

        st.pyplot(fig)


    def plot_average_correlation(lst, title_pos):
        # Create an empty DataFrame to store the combined data
        combined_dataset = pd.DataFrame()

        # Combine the data from all the years into one DataFrame
        for dataset, title in lst:
            combined_dataset = pd.concat([combined_dataset, dataset])

        ability_ratings = combined_dataset[
            ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
             'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
             'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
             'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
             'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
             'GKKicking', 'GKPositioning', 'GKReflexes']]
        overall_ratings = combined_dataset['Overall']

        # Calculate correlation between personal abilities and overall rating
        correlation_matrix = ability_ratings.corrwith(overall_ratings)

        # Sort the correlations in descending order
        sorted_correlations = correlation_matrix.abs().sort_values(ascending=False)

        # Plot the sorted correlations with reversed colors
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, color='steelblue')
        plt.xlabel('Correlation')
        plt.ylabel('Personal Ability')
        plt.title(f'{title_pos} Average Correlation between Personal Abilities and Overall Rating')

        # Add numbers on the bars
        for i, correlation in enumerate(sorted_correlations.values):
            plt.text(correlation, i, f'{correlation:.2f}', ha='left', va='center')

        plt.show()
        st.pyplot(plt)


    roles_lst = [(goalkeeper_lst, "Goalkeeper"), (defender_lst, "Defender"), (midfielder_lst, "Midfielder"),
                 (forward_lst, "Forward")]

    st.subheader("correlation between the player's individual ability ratings and his overall rating in the game")
    st.markdown("In this graph we get a numerical figure for the correlation between each feature and the Overall rating.")

    # Create a selectbox to choose the position
    position = st.selectbox("Select a position:", ["Goalkeeper", "Defender", "Midfielder", "Forward", "Averages"])

    if position == "Averages":
        plot_average_correlation(goalkeeper_lst, "Goalkeeper")
        plot_average_correlation(defender_lst, "Defender")
        plot_average_correlation(midfielder_lst, "Midfielder")
        plot_average_correlation(forward_lst, "Forward")
    else:
        # Find the corresponding list for the selected position
        lst = next((lst for lst, role in roles_lst if role == position), None)

        # If the list is found, plot the correlation
        if lst is not None:
            plot_correlation(lst, position)

# Option 3: The Dream Team
elif option == "The Dream Team":
    def find_dream_team(dataset, formation):
        # Remove rows with null values in the 'Position' column
        dataset = dataset.dropna(subset=['Position'])
        if formation == "433":
            # Define the core positions
            core_positions = ['GK', 'CB', 'RB', 'LB', 'CM', 'CDM', 'RW', 'LW', 'CF']
            double = ['CB', 'CM']
        elif formation == "442":
            core_positions = ['GK', 'CB', 'RB', 'LB', 'CM', 'RM', 'LM', 'ST']
            double = ['CB', 'CM', 'ST']
        elif formation == "4231":
            core_positions = ['GK', 'CB', 'RB', 'LB', 'CM', 'CDM', 'CAM', 'RW', 'LW', 'ST']
            double = ['CB']
        else:
            return {}

        # Initialize an empty dictionary to store the dream team
        dream_team = defaultdict(list)

        # Find the top player for each core position
        for position in core_positions:
            if position in double:
                top_players = dataset.loc[dataset['Best Position'].str.strip() == position].nlargest(2, 'Overall')
            else:
                top_players = dataset.loc[dataset['Best Position'].str.strip() == position].nlargest(1, 'Overall')
            if not top_players.empty:
                for i in range(len(top_players)):
                    player_name = top_players.iloc[i]['Name']
                    dream_team[position].append(player_name)

        return dream_team


    def plot_dream_team(dream_team, year, formation):
        pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
        fig, ax = pitch.draw()

        # Define the positions on the pitch for the dream team
        positions = {
            'GK': [(5, 40)],
            'CB': [(25, 30), (25, 50)],
            'RB': [(25, 70)],
            'LB': [(25, 5)],
            'CM': [(65, 30), (65, 50)],
            'CDM': [(40, 40)],
            'CAM': [(85, 40)],
            'RM': [(70, 70)],
            'LM': [(70, 5)],
            'RW': [(90, 70)],
            'LW': [(90, 5)],
            'CF': [(100, 40)],
            'ST': [(105, 35), (105, 45)]
        }

        # Plot the dream team players on the pitch
        for position, players in dream_team.items():
            for i, player_name in enumerate(players):
                coords = positions[position][i]
                ax.text(coords[0], coords[1], player_name, color='black', ha='center', va='center', fontsize=12,
                        fontweight='bold')

        plt.title(f"{year} Dream Team with Formation: {formation}")
        plt.show()

        st.pyplot(plt)


    games = ["FIFA20", "FIFA21", "FIFA22"]
    formations = ["433", "4231", "442"]

    st.subheader("Bulid your Dream Team according to the year and the formation")
    # Create selectboxes to choose the game and formation
    selected_game = st.selectbox("Select a game:", games)
    selected_formation = st.selectbox("Select a formation:", formations)

    # Find the corresponding dataset for the selected game
    if selected_game == "FIFA20":
        dataset = fifa20
    elif selected_game == "FIFA21":
        dataset = fifa21
    elif selected_game == "FIFA22":
        dataset = fifa22

    # If the dataset is not empty, find the dream team and plot it
    if not dataset.empty:
        dream_team = find_dream_team(dataset, selected_formation)
        plot_dream_team(dream_team, selected_game, selected_formation)
    else:
        st.write("No dataset available for the selected game.")

# Option 4: Player's potential in early years vs following years
elif option == "Player's potential":
    # Sorting the FIFA 17 dataset by the difference between Potential and Overall ratings in descending order and selecting top 20 players
    top_50_players_fifa17_defender = fifa17_defender.assign(
        Difference=fifa17_defender['Potential'] - fifa17_defender['Overall']).nlargest(50, 'Difference')
    top_50_players_fifa17_midfielder = fifa17_midfielder.assign(
        Difference=fifa17_midfielder['Potential'] - fifa17_midfielder['Overall']).nlargest(50, 'Difference')
    top_50_players_fifa17_forward = fifa17_forward.assign(
        Difference=fifa17_forward['Potential'] - fifa17_forward['Overall']).nlargest(50, 'Difference')
    top_50_players_fifa17_goalkeeper = fifa17_goalkeeper.assign(
        Difference=fifa17_goalkeeper['Potential'] - fifa17_goalkeeper['Overall']).nlargest(50, 'Difference')
    # Create empty DataFrames for each position
    result_df_defender = pd.DataFrame(columns=['ID', 'Name', 'Potential 17', 'Overall 17', 'Potential 18', 'Overall 18',
                                               'Potential 19', 'Overall 19', 'Potential 20', 'Overall 20',
                                               'Potential 21', 'Overall 21', 'Potential 22', 'Overall 22'])

    result_df_midfielder = pd.DataFrame(
        columns=['ID', 'Name', 'Potential 17', 'Overall 17', 'Potential 18', 'Overall 18',
                 'Potential 19', 'Overall 19', 'Potential 20', 'Overall 20',
                 'Potential 21', 'Overall 21', 'Potential 22', 'Overall 22'])

    result_df_forward = pd.DataFrame(columns=['ID', 'Name', 'Potential 17', 'Overall 17', 'Potential 18', 'Overall 18',
                                              'Potential 19', 'Overall 19', 'Potential 20', 'Overall 20',
                                              'Potential 21', 'Overall 21', 'Potential 22', 'Overall 22'])

    result_df_goalkeeper = pd.DataFrame(
        columns=['ID', 'Name', 'Potential 17', 'Overall 17', 'Potential 18', 'Overall 18',
                 'Potential 19', 'Overall 19', 'Potential 20', 'Overall 20',
                 'Potential 21', 'Overall 21', 'Potential 22', 'Overall 22'])

    lst = [(top_50_players_fifa17_defender, result_df_defender),
           (top_50_players_fifa17_midfielder, result_df_midfielder),
           (top_50_players_fifa17_forward, result_df_forward), (top_50_players_fifa17_goalkeeper, result_df_goalkeeper)]
    for top_50, df in lst:
        # Looping through the top 20 players and extracting their data from the respective datasets
        for index, player in top_50.iterrows():
            player_id = player['ID']
            player_name = player['Name']
            potential_17 = player['Potential']
            overall_17 = player['Overall']
            player_data = [player_id, player_name, potential_17, overall_17]

            # Extracting potential and overall values from FIFA 18 to FIFA 22
            for year, dataset in zip(['18', '19', '20', '21', '22'], [fifa18, fifa19, fifa20, fifa21, fifa22]):
                try:
                    player_row = dataset.loc[dataset['ID'] == player_id]
                    potential = player_row['Potential'].values[0]
                    overall = player_row['Overall'].values[0]
                except IndexError:
                    potential = None
                    overall = None

                # Add player data to the DataFrame only if Potential is higher than 75
                if potential is not None and potential > 75:
                    player_data.extend([potential, overall])
                else:
                    player_data.extend([None, None])

            # Add player data to the DataFrame only if Potential is higher than 75
            if player_data[2] is not None and player_data[2] > 75:
                df.loc[index] = player_data

    # Deleting rows with None values and keeping 20 rows in each dataset

    # Deleting rows with None values
    result_df_defender = result_df_defender.dropna()
    result_df_midfielder = result_df_midfielder.dropna()
    result_df_forward = result_df_forward.dropna()
    result_df_goalkeeper = result_df_goalkeeper.dropna()

    # Keeping 30 rows in each dataset
    result_df_defender = result_df_defender.head(30)
    result_df_midfielder = result_df_midfielder.head(30)
    result_df_forward = result_df_forward.head(30)
    result_df_goalkeeper = result_df_goalkeeper.head(30)

    def plot_summary(result_df, title):
        # Selecting the players from result_df
        players = result_df['Name'].tolist()

        # Sort players by their second name
        sorted_players = sorted(players, key=lambda x: x.split()[1])

        # Creating subplots for all players
        num_players = len(players)
        num_cols = 3
        num_rows = (num_players + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

        # Creating line plots for each player
        for i, player in enumerate(sorted_players):
            player_data = result_df[result_df['Name'] == player]

            years = ['FIFA 17', 'FIFA 18', 'FIFA 19', 'FIFA 20', 'FIFA 21', 'FIFA 22']
            potentials = player_data[['Potential 17', 'Potential 18', 'Potential 19', 'Potential 20', 'Potential 21',
                                      'Potential 22']].values[0]
            overalls = \
            player_data[['Overall 17', 'Overall 18', 'Overall 19', 'Overall 20', 'Overall 21', 'Overall 22']].values[0]

            row = i // num_cols
            col = i % num_cols

            ax = axes[row, col]
            ax.plot(years, potentials, marker='o', label='Potential')
            ax.plot(years, overalls, marker='o', label='Overall')

            ax.set_title(f"Player: {player}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Rating")

        # Adding main title
        fig.suptitle(title, fontsize=16, y=1.05)

        # Creating a single legend for all plots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')

        # Adjusting the layout and spacing
        plt.tight_layout()

        # Displaying the figure
        plt.show()
        st.pyplot(plt)

    st.subheader("Is the player's potential in the early years realized in the following years?")
    st.markdown("In this graph we can see the change trend of the Potential and Overall ratings over the years, and conclude whether the player has realized his Potential.")
    position = st.selectbox("Select Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"])

    # Assign the appropriate list based on the selected position
    if position == "Goalkeeper":
        plot_summary(result_df_goalkeeper, "Goalkeeper Position")
    elif position == "Defender":
        plot_summary(result_df_defender, "Defender Position")
    elif position == "Midfielder":
        plot_summary(result_df_midfielder, "Midfielder Position")
    elif position == "Forward":
        plot_summary(result_df_forward, "Forward Position")

# option 5 - "'Nationality Distribution among Top Players'"
elif option == "Nationality Distribution among Top Players":
    def plot_player_distribution(dataset, n):
        # Load the dataset
        df = pd.read_csv(dataset)

        # Select the top n players by "Overall" attribute
        top_players = df.nlargest(n, "Overall")

        # Count the number of players with each nationality
        country_counts = top_players['Nationality'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']

        # Plot the world map with colored countries
        fig = px.choropleth(country_counts, locations='Country', locationmode='country names',
                            color='Count',
                            hover_name='Country', color_continuous_scale='thermal')

        # Set the map projection to 'natural earth'
        fig.update_geos(projection_type="natural earth")

        # Display the plot
        st.plotly_chart(fig)


    # Set the title of the app
    st.subheader('Nationality Distribution among Top Players')

    # Define the dataset dictionary
    datasets = {
        'fifa17': {
            'Goalkeeper': 'fifa17_goalkeeper.csv',
            'Defender': 'fifa17_defender.csv',
            'Midfielder': 'fifa17_midfielder.csv',
            'Forward': 'fifa17_forward.csv',
            'All Positions': 'fifa17.csv'
        },
        'fifa18': {
            'Goalkeeper': 'fifa18_goalkeeper.csv',
            'Defender': 'fifa18_defender.csv',
            'Midfielder': 'fifa18_midfielder.csv',
            'Forward': 'fifa18_forward.csv',
            'All Positions': 'fifa18.csv'
        },
        'fifa19': {
            'Goalkeeper': 'fifa19_goalkeeper.csv',
            'Defender': 'fifa19_defender.csv',
            'Midfielder': 'fifa19_midfielder.csv',
            'Forward': 'fifa19_forward.csv',
            'All Positions': 'fifa19.csv'
        },
        'fifa20': {
            'Goalkeeper': 'fifa20_goalkeeper.csv',
            'Defender': 'fifa20_defender.csv',
            'Midfielder': 'fifa20_midfielder.csv',
            'Forward': 'fifa20_forward.csv',
            'All Positions': 'fifa20.csv'
        },
        'fifa21': {
            'Goalkeeper': 'fifa21_goalkeeper.csv',
            'Defender': 'fifa21_defender.csv',
            'Midfielder': 'fifa21_midfielder.csv',
            'Forward': 'fifa21_forward.csv',
            'All Positions': 'fifa21.csv'
        },
        'fifa22': {
            'Goalkeeper': 'fifa22_goalkeeper.csv',
            'Defender': 'fifa22_defender.csv',
            'Midfielder': 'fifa22_midfielder.csv',
            'Forward': 'fifa22_forward.csv',
            'All Positions': 'fifa22.csv'
        }
    }

    # Dataset selection
    selected_dataset = st.selectbox('Select a dataset', options=list(datasets.keys()))

    # Position selection
    positions = list(datasets[selected_dataset].keys())
    selected_position = st.selectbox('Select a position', options=positions)

    n_players = st.slider('Select the number of top players', min_value=10, max_value=500, value=100, step=10)

    # Get the dataset filename based on the selected dataset and position
    dataset_file = datasets[selected_dataset][selected_position]

    # Plot the player distribution on a world map
    plot_player_distribution(dataset_file, n_players)

# option 6 - "Who is better in FIFA, Messi or Ronaldo?"
elif option == "Who is better in FIFA, Messi or Ronaldo?":
    fifa22.loc[fifa22['ID'] == 20801, 'Name'] = ' Cristiano Ronaldo'
    fifa22.loc[fifa22['ID'] == 158023, 'Name'] = ' L. Messi'

    def calculate_average_correlation(lst):
        all_sorted_correlations = []

        for dataset, title in lst:
            ability_ratings = dataset[
                ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
                 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                 'GKKicking', 'GKPositioning', 'GKReflexes']]
            overall_ratings = dataset['Overall']

            # Calculate correlation between personal abilities and overall rating
            correlation_matrix = ability_ratings.corrwith(overall_ratings)

            # Sort the correlations in descending order
            sorted_correlations = correlation_matrix.abs().sort_values(ascending=False)

            return sorted_correlations


    abilities_to_check = calculate_average_correlation(forward_lst).to_dict()
    top_abilities = list(abilities_to_check.keys())[:11]
    top_abilities.insert(0, "Overall")

    # Initialize empty lists for each player's ratings
    messi_ratings = ronaldo_ratings = defaultdict(list)
    ronaldo_ratings = {ability: [] for ability in top_abilities}
    messi_ratings = {ability: [] for ability in top_abilities}

    # Check if Messi and Ronaldo exist in each dataset and extract their ratings for each feature
    datasets = [fifa17, fifa18, fifa19, fifa20, fifa21, fifa22]
    features = top_abilities

    for dataset in datasets:
        for feature in features:
            messi_ratings[feature].append(dataset.loc[dataset['Name'] == ' L. Messi', feature].values[0])
            ronaldo_ratings[feature].append(dataset.loc[dataset['Name'] == ' Cristiano Ronaldo', feature].values[0])

    years = ['FIFA 17', 'FIFA 18', 'FIFA 19', 'FIFA 20', 'FIFA 21', 'FIFA 22']

    # Create subplots for each feature
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
    plt.suptitle('Lionel Messi vs Cristiano Ronaldo - Ratings Comparison')

    # Plot each feature on a separate subplot
    for i, ax in enumerate(axes.flat):
        feature = features[i]
        ax.plot(years, messi_ratings[feature], marker='o', label='Messi')
        ax.plot(years, ronaldo_ratings[feature], marker='o', label='Ronaldo')
        ax.set_title(feature)
        ax.set_xlabel('FIFA Edition')
        ax.set_ylabel('Rating')
        ax.legend()

        # Set y-axis limits and ticks
        ax.set_ylim(70, 100)
        ax.set_yticks(range(70, 101, 2))

    # Adjust the layout and spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show the plot
    st.subheader("Who is better in FIFA, Messi or Ronaldo?")
    st.markdown("In this graph we see a comparison between Messi and Ronaldo in the features that have the highest correlation with the Overall rating among forwards over the years.")
    st.pyplot(plt)


