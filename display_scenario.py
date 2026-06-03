import os
import cv2
import numpy as np
import random

# Load the image table.png


def create_scenario_image(board1, board2, herocards, opponentcardslist, display=True):
    image = cv2.imread('images/table.png')
    # Rotate the image by 90 degrees
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    classesfolder = "C:/PycharmProjects/progressiveautolabelling/cropped"
    classes = os.listdir(classesfolder)    # The format of the game is double board PLO
    # The board1 is a list of cards like ['As','Ks','Qs']
    # The board2 is a list of cards like ['Ad','Kd','Qd']
    # The herocards is a list of dicts with keys 'cards' and 'equity'
    # The opponentcardslist is a list of dicts with keys 'cards' and 'equity'
    # The function will create a scenario image with the given parameters
    # The function will return the image as cv2 image
    # If display is true, the function will display the image

    # Create a blank image
    scenario_image = image.copy()
    # Make it 150% bigger
    scenario_image = cv2.resize(scenario_image, (int(scenario_image.shape[1] * 1.5), int(scenario_image.shape[0] * 1.5)))

    # Load card images
    card_images = {}
    for card in board1 + board2 + [card for sublist in herocards for card in sublist['cards']] + [card for sublist in opponentcardslist for card in sublist['cards']]:
        card_image_path = os.path.join(classesfolder, card, max(os.listdir(os.path.join(classesfolder, card)), key=lambda x: os.path.getsize(os.path.join(classesfolder, card, x))))
        card_image = cv2.imread(card_image_path)
        if card_image is None:
            print(f"Warning: Could not read image for card {card} at path {card_image_path}")
        else:
            card_images[card] = card_image

    opponent_gap_from_side_left = 5
    opponent_gap_from_side_right = 60
    opponent_gap_from_top = 40

    side_opponent_gap_from_top = 20

    hero_gap_from_bottom = 10
    hero_gap_from_side = 35

    board1_gap_from_top = 35
    board_gap_from_side = 40

    board2_gap_from_top = 50

    card_size = 5

    # Define positions in percentages
    positions = {
        'board1': [(board_gap_from_side, board1_gap_from_top, card_size)],
        'board2': [(board_gap_from_side, board2_gap_from_top, card_size)],
        'hero': [(hero_gap_from_side, 100 - hero_gap_from_bottom, card_size)],
        'opponents': [
            [(hero_gap_from_side, hero_gap_from_bottom, card_size)],  # Opposite to hero
            [(opponent_gap_from_side_left, side_opponent_gap_from_top, card_size)],  # Left side top
            [(opponent_gap_from_side_left, 100 - opponent_gap_from_top, card_size)],  # Left side bottom
            [(opponent_gap_from_side_right, side_opponent_gap_from_top, card_size)],  # Right side top
            [(opponent_gap_from_side_right, 100 - opponent_gap_from_top, card_size)]  # Right side bottom
        ]
    }

    # Randomly assign seats to opponents
    opponent_seats = random.sample(positions['opponents'], len(opponentcardslist) + len(herocards) - 1)

    # Function to place cards on the image
    def place_cards(cards, start_position, scenario_image, equity=None):
        x_percent, y_percent, size_percent = start_position[0]
        x = int(x_percent / 100 * scenario_image.shape[1])
        y = int(y_percent / 100 * scenario_image.shape[0])
        card_width = int(size_percent / 100 * scenario_image.shape[1])
        horizontal_spacing = int(0.5 / 100 * scenario_image.shape[1])
        text_height = int(2 / 100 * scenario_image.shape[0])

        if equity is not None:
            equity_text = equity
            text_size = cv2.getTextSize(equity_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x
            text_y = y - text_height
            cv2.putText(scenario_image, equity_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_height / text_size[1], (255, 255, 255), 2, cv2.LINE_AA)

        for i, card in enumerate(cards):
            if card_images[card] is None:
                continue
            card_image = card_images[card]
            aspect_ratio = card_image.shape[1] / card_image.shape[0]
            card_height = int(card_width / aspect_ratio)
            resized_card = cv2.resize(card_image, (card_width, card_height))
            if y + card_height <= scenario_image.shape[0] and x + card_width <= scenario_image.shape[1]:
                scenario_image[y:y + card_height, x:x + card_width] = resized_card
            x += card_width + horizontal_spacing

    # Place board1 cards
    place_cards(board1, positions['board1'], scenario_image)

    # Place board2 cards
    place_cards(board2, positions['board2'], scenario_image)

    # Place hero cards
    place_cards(herocards[0]['cards'], positions['hero'], scenario_image, herocards[0].get('equity'))

    # Place opponent cards
    for j, opponent in enumerate(opponentcardslist):
        place_cards(opponent['cards'], opponent_seats[j], scenario_image, opponent.get('equity'))

    for k, hero in enumerate(herocards[1:]):
        place_cards(hero['cards'], opponent_seats[len(opponentcardslist) + k], scenario_image, hero.get('equity'))

    # Display the image if required
    if display:
        cv2.imshow('Scenario Image', scenario_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scenario_image

if __name__ == '__main__':
    # Test
    board1 = ['As', 'Ks', 'Qs']
    board2 = ['Ad', 'Kd', 'Qd']
    herocards = [{'cards': ['Ac', 'Kc', 'Qc', 'Jc'], 'equity': "0.5, 0.8"}]
    opponentcardslist = [{'cards': ['2s', '3s', '4s', '5s'], 'equity': "0.25"}, {'cards': ['2d', '3d', '4d', '5d'], 'equity': "0.25"}]
    create_scenario_image(board1, board2, herocards, opponentcardslist, display=True)