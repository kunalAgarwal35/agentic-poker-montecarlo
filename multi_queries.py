from itertools import combinations
import os
import subprocess
import logging
from pprint import pprint
import threading
import time
import pickle
import itertools
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# ============= JAVA JVM WARMING =============
_JVM_WARMED = False

def warmup_jvm():
    """
    Pre-warm the JVM by running a simple query.
    This loads classes and JIT compiles hot paths, making subsequent calls faster.
    Call this at server startup (e.g., in health check).
    """
    global _JVM_WARMED
    if _JVM_WARMED:
        print("[JVM] Already warmed up")
        return
    
    try:
        print("[JVM] Warming up Java process...")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java_files")
        # Simple query to warm up JVM
        warmup_cmd = f'java.exe -XX:+TieredCompilation -XX:TieredStopAtLevel=1 -Xshare:auto -cp {path}\\p2.jar propokertools.cli.RunPQL -mt 100 -ms 5 "select count(*) from game=\'holdem\', PLAYER_1=\'AA\', PLAYER_2=\'KK\'"'
        subprocess.check_output(warmup_cmd, shell=True, timeout=10)
        _JVM_WARMED = True
        print("[JVM] Warmup complete")
    except Exception as e:
        print(f"[JVM] Warmup failed (non-critical): {e}")
        _JVM_WARMED = True  # Don't retry on failure
# =============================================


class PlayerWithEquity(object):

    def __init__(self, logger):
        """
        Initializing the class
        """
        self.verbose_mode = False
        self.headers = {"Host": "www.propokertools.com",
                        "Connection": "keep-alive",
                        "Accept": "text/javascript, text/html, application/xml, text/xml, */*",
                        "X-Prototype-Version": "1.6.0.3",
                        "DNT": "1",
                        "X-Requested-With": "XMLHttpRequest",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                        "Content-type": "application/x-www-form-urlencoded; charset=UTF-8",
                        "Origin": "http://www.propokertools.com",
                        "Referer": "http://www.propokertools.com/pql",
                        "Accept-Encoding": "gzip, deflate", }


        self.num_iterations = 30000  # Default Java iterations (can be reduced for speed)
        self.max_time = 120  # Max seconds for Java query
        self.fast_mode = False  # Set to True for faster but less accurate results
        self.ev_filter = '15'
        self.ev_filter_switch = 1
        self.loh15 = pickle.load(open("loh15.pkl", "rb"))
        self.loh10 = pickle.load(open("loh10.pkl", "rb"))

        self.cards = set()
        self.cards_to_run_for = dict()
        self.logger = logger

        self.default_range = "25%"

        self.num_players = 6

        self.modes = [
            1,  # EV
            2,  # Best hand and Range
            3,  # Both
        ]
        self.mode = 3
        self.disabled_sources = set()
        self.path_to_jar_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "java_files"
        )
        self.online = False
        self.timeout = 30
        self.unknown_range_1 = "100%"
        self.unknown_range_2 = "100%"
        self.board = set()
        self.game_type = None
        # C:\\oracle_java\\bin\\
        # Optimized JVM flags: -XX:+TieredCompilation -XX:TieredStopAtLevel=1 for faster startup
        # -Xshare:auto uses class data sharing if available
        self.command = """java.exe -XX:+TieredCompilation -XX:TieredStopAtLevel=1 -Xshare:auto -cp {folderpath}\\p2.jar propokertools.cli.RunPQL  -mt {max_iterations} -ms {max_seconds} "{query}"
        """


    def format_cards(self, cards):
        """
        Format cards which are to be printed
        """
        if isinstance(cards, list):
            for i, card in enumerate(cards):
                cards[i] = card.capitalize()
            return cards
        else:
            cards = list(cards)
            for i in range(0, len(cards), 2):
                cards[i] = cards[i].upper()
            return "".join(cards)


    def reset_variables(self, **kwargs):
        """
        Reset all the variables to their default values. Typically this is done at the end of each iteration
        """
        self.cards = set()
        self.board = set()
        self.cards_to_run_for = dict()
        self.unknown_range_1 = kwargs.get("range", self.default_range)
        self.unknown_range_2 = kwargs.get("range", self.default_range)


    def add_cards(self, *args):
        """
        Adding cards to the object
        """
        for arg in args:
            self.cards.add(
                self.format_cards(arg)
            )
            self.cards_to_run_for[self.format_cards(arg)] = True

    def add_to_board(self, *args):
        """
        Adding cards to the board
        """
        for arg in args:
            self.board.add(
                self.format_cards(arg)
            )

    def set_max_time(self, timeout):
        """
        Setting requests timeout for the table
        """
        self.timeout = timeout

    def set_mode(self, mode):
        """
        Setting the default run mode.
        Mode can be one of the three
        1. EV - "ev" - Running only the EV query
        2. Best hand and Range - "od" - Running both the EV and Best hand and Range queries
        3. Both - "all" - Running both the EV and Best hand and Range queries
        """
        if mode == "ev":
            self.mode = 1
            return mode
        elif mode == "od":
            self.mode = 2
            return mode
        elif mode == "all":
            self.mode = 3
            return mode
        else:
            return False

    def set_game_type(self, num_cards):
        """
        Sets the type of the game. This mostly runs automatically by detecting the number of cards but can be set manually
        """
        if num_cards == 4 or num_cards == 8:
            self.game_type = 4
        elif num_cards == 5 or num_cards == 10:
            self.game_type = 5
        elif num_cards == 6 or num_cards == 12:
            self.game_type = 6



    def raw_ev_query(self, cards1, cards2, dictionary, pop_dict, **kwargs):
        """
        Raw EV query function to run EV query and append to the input dictionary
        KW Args:
            board: Board cards
            deadcards: Dead cards
            online: Boolean to run online or locally
        """
        board = kwargs.get("board", "")
        online = kwargs.get("online", False)
        deadcards = kwargs.get("deadcards", "")
        # if self.game_type is None:
        if len(cards1) == 10:
            self.set_game_type(10)
            tablename = "omahahi5"
        elif len(cards1) == 8:
            self.set_game_type(8)
            tablename = "omahahi"
        elif len(cards1) == 12:
            self.set_game_type(12)
            tablename = "omahahi6"
        # else:
        #     if self.game_type == 4:
        #         tablename = "omahahi"
        #     elif self.game_type == 6:
        #         tablename = "omahahi6"
        #     else:
        #         tablename = "omahahi5"

        query = """        select /* Start equity stats */            avg(riverEquity(PLAYER_1)) as PLAYER_1_equity1,            count(winsHi(PLAYER_1)) as PLAYER_1_winsHi1,            count(tiesHi(PLAYER_1)) as PLAYER_1_tiesHi1,            avg(riverEquity(PLAYER_2)) as PLAYER_2_equity1,            count(winsHi(PLAYER_2)) as PLAYER_2_winsHi1,            count(tiesHi(PLAYER_2)) as PLAYER_2_tiesHi1        /* End equity stats */        from game='{tablename}', syntax='Generic',            dead='{deadcards}',            board='{board}',            PLAYER_1='{cards1}',            PLAYER_2='{cards2}'        """.format(
            tablename=tablename,
            deadcards=deadcards,
            board=board,
            cards1=cards1,
            cards2=cards2
        )
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )
        if online:
            res = self.run_query_online(query)
            p1_equity = float(res["PLAYER_1_EQUITY1"])
            p2_equity = float(res["PLAYER_2_EQUITY1"])
        else:
            res = subprocess.check_output(
                query_string).decode('utf8').split('\n')
            p1_equity = float(res[0].split(" = ")[1])
            p2_equity = float(res[3].split(" = ")[1])
        dictionary["{}_{}".format(cards1, cards2)] = {
            "p1": p1_equity,
            "p2": p2_equity
        }
        pop_dict["{}_{}".format(cards1, cards2)] = p2_equity

    def xmyz(self,h):
        cards = [h[:2], h[2:4], h[4:6], h[6:8], h[8:10]]
        suits = [h[1], h[3], h[5], h[7], h[9]]
        suits = list(set(suits))
        new_suits = ['x', 'y', 'z', 'm'][0:len(suits)]
        new_cards = []
        for c in cards:
            new_cards.append(c[0] + new_suits[suits.index(c[1])])
        new_cards = sorted(new_cards)
        new_cards = ''.join(new_cards)
        return new_cards

    def check_ev_quality(self,cards1,cards2):
        """
        Checks if either cards 1 or cards 2 falls out of the ev filter range

        :param cards1:
        :param cards2:

        """
        if self.ev_filter == '100':
            return True

        ranks = ['A','K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        cards1 = [cards1[0:2],cards1[2:4],cards1[4:6],cards1[6:8],cards1[8:10]]
        cards2 = [cards2[0:2],cards2[2:4],cards2[4:6],cards2[6:8],cards2[8:10]]
        cards1 = sorted(cards1, key=lambda x: ranks.index(x[0]))
        cards2 = sorted(cards2, key=lambda x: ranks.index(x[0]))
        cards1 = ''.join(cards1)
        cards2 = ''.join(cards2)
        cards1 = self.xmyz(cards1)
        cards2 = self.xmyz(cards2)

        if self.ev_filter == '15':
            if cards1 not in self.loh15 and cards2 not in self.loh15:
                return False
            else:
                return True
        elif self.ev_filter == '10':
            if cards1 not in self.loh10 and cards2 not in self.loh10:
                return False
            else:
                return True



    def run_ev_query_for_multiple_cards(self, *args, **kwargs):
        """
        Running the EV query in threaded mode. This is the main function to run the EV query.
        Args:
            All the different holecards. This should be greater than 3
        KW Args:
            board: Board cards
        Returns the edge of knowing deadcards versus without knowing deadcards, for board as well as without board cards
        """
        if len(args) < 3:
            raise Exception("Minimum number of cards should be 3")

        board = kwargs.get("board", "")
        with_deadcards = {}
        without_deadcards = {}
        pop_with_deadcards = {}
        pop_without_deadcards = {}
        threads = []
        # print("Length of combinations: {}".format(
        #     len(list(combinations(args, 2)))))
        for cards1, cards2 in combinations(args, 2):
            if not self.cards_to_run_for[cards1] and not self.cards_to_run_for[cards2]:
                # print("Skipping {} and {} | {}".format(cards1, cards2, s))
                continue
            if len(board) == 0 and not self.check_ev_quality(cards1,cards2):
                continue
            # Without deadcards
            t = threading.Thread(
                target=self.raw_ev_query,
                args=(cards1, cards2, without_deadcards, pop_without_deadcards),
                kwargs=dict(board=board, online=self.online))
            threads.append(t)
            t.start()

            # With deadcards
            remaining_cards = list(args)
            remaining_cards.pop(remaining_cards.index(cards1))
            remaining_cards.pop(remaining_cards.index(cards2))
            t2 = threading.Thread(
                target=self.raw_ev_query,
                args=(cards1, cards2, with_deadcards, pop_with_deadcards),
                kwargs=dict(board=board, deadcards="".join(remaining_cards), online=self.online))
            threads.append(t2)
            t2.start()

        for thread in threads:
            thread.join()

        del threads

        keys = with_deadcards.keys()
        ret_dict = {}
        ret_pop_cards = {}
        for _, key in enumerate(keys):
            cards1, cards2 = key.split("_")
            ret_dict["{} : {}".format(cards1, cards2)] = 100 * (
                0.97 * float(without_deadcards[key]["p1"])
                - float(with_deadcards[key]["p1"])
            )

            ret_dict["{} : {}".format(cards2, cards1)] = 100 * (
                0.97 * float(without_deadcards[key]["p2"])
                - float(with_deadcards[key]["p2"]))
            ret_pop_cards["{} : {}".format(
                cards1, cards2)] = pop_with_deadcards[key]
            ret_pop_cards["{} : {}".format(
                cards2, cards1)] = 1 - pop_with_deadcards[key]
        positive_only = {}
        for _, value in ret_dict.items():
            if value > 0:
                positive_only[_] = value
        del ret_dict

        return positive_only, ret_pop_cards

    def raw_best_hand_query(self, dictionary, hero_cards, dead_cards, p2_range, board="", online=False):
        """
        Raw Best Hand query function.
        Args:
            dictionary: Dictionary to append the results to
            hero_cards: Hero cards
            dead_cards: Dead cards
            p2_range: Range of P2
            board: Board cards
            online: Boolean to run online or locally
        """
        # if self.game_type is None:
        if len(hero_cards) == 10:
            self.set_game_type(10)
            tablename = "omahahi5"
        elif len(hero_cards) == 8:
            self.set_game_type(8)
            tablename = "omahahi"
        elif len(hero_cards) == 12:
            self.set_game_type(12)
            tablename = "omahahi6"
            p2_range = "AAKK,KQTJ"
        # else:
        #     if self.game_type == 4:
        #         tablename = "omahahi"
        #     elif self.game_type == 6:
        #         tablename = "omahahi6"
        #     else:
        #         tablename = "omahahi5"

        query = """        select /* Start equity stats */        avg(riverEquity(PLAYER_1)) as PLAYER_1_equity1,            count(winsHi(PLAYER_1)) as PLAYER_1_winsHi1,            count(tiesHi(PLAYER_1)) as PLAYER_1_tiesHi1,            avg(riverEquity(PLAYER_2)) as PLAYER_2_equity1,            count(winsHi(PLAYER_2)) as PLAYER_2_winsHi1,            count(tiesHi(PLAYER_2)) as PLAYER_2_tiesHi1        /* End equity stats */        from game='{tablename}', syntax='Generic',            board='{board}',            dead='{dead_cards}',            PLAYER_1='{hero_cards}',            PLAYER_2='{p2_value}'        """.format(
            tablename=tablename, board=board, dead_cards=dead_cards, hero_cards=hero_cards, p2_value=p2_range)
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )
        if online:
            res = self.run_query_online(query)
        else:
            resp = subprocess.check_output(query_string).decode('utf8')
            lines = resp.split("\n")
        try:
            if online:
                p1_equity = float(res["PLAYER_1_EQUITY1"]) * 100
            else:
                p1_equity = float(lines[0].split(" = ")[1][:6]) * 100
        except ValueError:
            p1_equity = "Could not calculate"
        except IndexError:
            if online:
                print("Internet connection error")
                p1_equity = "Internet connection error"
            else:
                p1_equity = "Invalid data output: {}".format(lines[0])
        dictionary[hero_cards] = p1_equity
        # print("Card:", hero_cards, ":", p1_equity)

    def equity_with_dc(self, hero_cards, dead_cards, p2_range, board=""):
        """
        Raw Best Hand query function.
        Args:
            dictionary: Dictionary to append the results to
            hero_cards: Hero cards
            dead_cards: Dead cards
            p2_range: Range of P2
            board: Board cards
            online: Boolean to run online or locally
        """
        # if self.game_type is None:
        dictionary = {}
        if len(hero_cards) == 10:
            self.set_game_type(10)
            tablename = "omahahi5"
        elif len(hero_cards) == 8:
            self.set_game_type(8)
            tablename = "omahahi"
        elif len(hero_cards) == 12:
            self.set_game_type(12)
            tablename = "omahahi6"
            p2_range = "AAKK,KQTJ"
        else:
            print("Invalid number of cards: ", hero_cards)

        query = """        select /* Start equity stats */        avg(riverEquity(PLAYER_1)) as PLAYER_1_equity1,            count(winsHi(PLAYER_1)) as PLAYER_1_winsHi1,            count(tiesHi(PLAYER_1)) as PLAYER_1_tiesHi1,            avg(riverEquity(PLAYER_2)) as PLAYER_2_equity1,            count(winsHi(PLAYER_2)) as PLAYER_2_winsHi1,            count(tiesHi(PLAYER_2)) as PLAYER_2_tiesHi1        /* End equity stats */        from game='{tablename}', syntax='Generic',            board='{board}',            dead='{dead_cards}',            PLAYER_1='{hero_cards}',            PLAYER_2='{p2_value}'        """.format(
            tablename=tablename, board=board, dead_cards=dead_cards, hero_cards=hero_cards, p2_value=p2_range)
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )

        resp = subprocess.check_output(query_string).decode('utf8')
        lines = resp.split("\n")
        try:
            p1_equity = float(lines[0].split(" = ")[1][:6]) * 100
        except ValueError:
            p1_equity = "Could not calculate"
        except IndexError:
            p1_equity = "Invalid data output: {}".format(lines[0])
        dictionary[hero_cards] = p1_equity
        # print("Card:", hero_cards, ":", p1_equity)
        return dictionary

    def pairwise_with_dc(self, player_range, p2_cards, p3_cards, deadcards, board=""):
        """
        Raw P1 Range query function.
        Args:
            player_range: Range of P1
            p2_cards: P2 cards
            p3_cards: P3 cards
            deadcards: Dead cards
            board: Board cards
            dictionary: Dictionary to append the results to
            online: Boolean to run online or locally
        This function runs in conjugation with Best Hand Query (raw_best_hand_query) to make up "od" mode
        """
        # if self.game_type is None:
        dictionary = {}
        if len(p2_cards) == 10:
            self.set_game_type(10)
            tablename = "omahahi5"
        elif len(p2_cards) == 8:
            self.set_game_type(8)
            tablename = "omahahi"
        elif len(p2_cards) == 12:
            self.set_game_type(12)
            tablename = "omahahi6"
            player_range = "AAKK,KQTJ"
        query = """    select /* Start equity stats */    avg(riverEquity(PLAYER_1)) as PLAYER_1_equity1,    count(winsHi(PLAYER_1)) as PLAYER_1_winsHi1,    count(tiesHi(PLAYER_1)) as PLAYER_1_tiesHi1,    avg(riverEquity(PLAYER_2)) as PLAYER_2_equity1,    count(winsHi(PLAYER_2)) as PLAYER_2_winsHi1,    count(tiesHi(PLAYER_2)) as PLAYER_2_tiesHi1,    avg(riverEquity(PLAYER_3)) as PLAYER_3_equity1,    count(winsHi(PLAYER_3)) as PLAYER_3_winsHi1,    count(tiesHi(PLAYER_3)) as PLAYER_3_tiesHi1    /* End equity stats */ from game='{tablename}', syntax='Generic',    dead='{deadcards}',    board='{board}',    PLAYER_1='{p1_range}',    PLAYER_2='{p2_cards}',    PLAYER_3='{p3_cards}'""".format(
            tablename=tablename,
            deadcards=deadcards,
            board=board,
            p1_range=player_range,
            p2_cards=p2_cards,
            p3_cards=p3_cards
        )
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )
        # print(query_string)
        # exit()

        resp = subprocess.check_output(query_string).decode('utf8')
        # print(resp)
        lines = resp.split("\n")
        try:
            unknown_range_eq = float(lines[0].split(" = ")[1]) * 100
        except ValueError:
            unknown_range_eq = "RAW: {}".format(lines[0])
        except IndexError:
            unknown_range_eq = "Invalid output on console"

        try:
            player_2_equity = float(lines[3].split(" = ")[1]) * 100
        except ValueError:
            player_2_equity = "RAW: {}".format(lines[3])
        except IndexError:
            player_2_equity = "Invalid output on console"

        try:
            player_3_equity = float(lines[6].split(" = ")[1]) * 100
        except ValueError:
            player_3_equity = "RAW: {}".format(lines[6])
        except IndexError:
            player_3_equity = "Invalid output on console"


        dictionary["{} : {}".format(p2_cards, p3_cards)] = {
            player_range: unknown_range_eq,
            p2_cards: player_2_equity,
            p3_cards: player_3_equity
        }
        return dictionary


    def run_best_hand_query_for_multiple_hands(self, *args, **kwargs):
        """
        Running the best hand query for all the cards. This is the main function to run the best hand query. Runs in threaded mode
        Args:
            All the different holecards. This should be greater than 3
        KW Args:
            board: Board cards
            unknown_range: Range of unknown cards
            online: Boolean to run online or locally

        Returns the dictionary of all the cards and their equity
        """
        board = kwargs.get("board", "")
        unknown_range = kwargs.get("unknown_range", "")
        main_dictionary = {}
        threads = []
        online = kwargs.get("online", False)
        for p1_cards in args:
            remaining_cards = list(args)
            remaining_cards.pop(remaining_cards.index(p1_cards))
            t1 = threading.Thread(
                target=self.raw_best_hand_query,
                args=(main_dictionary, p1_cards, "".join(remaining_cards),
                      unknown_range, board, online)
            )
            threads.append(t1)
            t1.start()

        for thread in threads:
            thread.join()

        del threads
        main_dictionary["metadata"] = "Using range - {}".format(
            unknown_range)
        return main_dictionary

    def raw_p1_range_query(self, player_range, p2_cards, p3_cards, deadcards, board="", dictionary={}, online=False):
        """
        Raw P1 Range query function.
        Args:
            player_range: Range of P1
            p2_cards: P2 cards
            p3_cards: P3 cards
            deadcards: Dead cards
            board: Board cards
            dictionary: Dictionary to append the results to
            online: Boolean to run online or locally
        This function runs in conjugation with Best Hand Query (raw_best_hand_query) to make up "od" mode
        """
        # if self.game_type is None:
        if len(p2_cards) == 10:
            self.set_game_type(10)
            tablename = "omahahi5"
        elif len(p2_cards) == 8:
            self.set_game_type(8)
            tablename = "omahahi"
        elif len(p2_cards) == 12:
            self.set_game_type(12)
            tablename = "omahahi6"
            player_range = "AAKK,KQTJ"
        query = """    select /* Start equity stats */    avg(riverEquity(PLAYER_1)) as PLAYER_1_equity1,    count(winsHi(PLAYER_1)) as PLAYER_1_winsHi1,    count(tiesHi(PLAYER_1)) as PLAYER_1_tiesHi1,    avg(riverEquity(PLAYER_2)) as PLAYER_2_equity1,    count(winsHi(PLAYER_2)) as PLAYER_2_winsHi1,    count(tiesHi(PLAYER_2)) as PLAYER_2_tiesHi1,    avg(riverEquity(PLAYER_3)) as PLAYER_3_equity1,    count(winsHi(PLAYER_3)) as PLAYER_3_winsHi1,    count(tiesHi(PLAYER_3)) as PLAYER_3_tiesHi1    /* End equity stats */ from game='{tablename}', syntax='Generic',    dead='{deadcards}',    board='{board}',    PLAYER_1='{p1_range}',    PLAYER_2='{p2_cards}',    PLAYER_3='{p3_cards}'""".format(
            tablename=tablename,
            deadcards=deadcards,
            board=board,
            p1_range=player_range,
            p2_cards=p2_cards,
            p3_cards=p3_cards
        )
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )
        # print(query_string)
        # exit()
        if not online:
            resp = subprocess.check_output(query_string).decode('utf8')
            # print(resp)
            lines = resp.split("\n")
            try:
                unknown_range_eq = float(lines[0].split(" = ")[1]) * 100
            except ValueError:
                unknown_range_eq = "RAW: {}".format(lines[0])
            except IndexError:
                unknown_range_eq = "Invalid output on console"

            try:
                player_2_equity = float(lines[3].split(" = ")[1]) * 100
            except ValueError:
                player_2_equity = "RAW: {}".format(lines[3])
            except IndexError:
                player_2_equity = "Invalid output on console"

            try:
                player_3_equity = float(lines[6].split(" = ")[1]) * 100
            except ValueError:
                player_3_equity = "RAW: {}".format(lines[6])
            except IndexError:
                player_3_equity = "Invalid output on console"
        else:
            resp = self.run_query_online(query)
            unknown_range_eq = float(resp["PLAYER_1_EQUITY1"]) * 100
            player_2_equity = float(resp["PLAYER_2_EQUITY1"]) * 100
            player_3_equity = float(resp["PLAYER_3_EQUITY1"]) * 100

        dictionary["{} : {}".format(p2_cards, p3_cards)] = {
            player_range: unknown_range_eq,
            p2_cards: player_2_equity,
            p3_cards: player_3_equity
        }

    def run_p1_range_query_for_multiple_hands(self, *args, **kwargs):
        """
        Running the P1 Range query for all the cards. This is the main function to run the P1 Range query. Runs in threaded mode
        Args:
            All the different holecards. This should be greater than 3
        KW Args:
            board: Board cards
            unknown_range: Range of unknown cards
            online: Boolean to run online or locally
        Returns the dictionary of all the cards and their equity
        """
        board = kwargs.get("board", "")
        unknown_range = kwargs.get("unknown_range", "")
        main_dictionary = {}
        threads = []
        online = kwargs.get("online", False)
        for p2_cards, p3_cards in combinations(args, 2):

            remaining_cards = list(args)
            remaining_cards.pop(remaining_cards.index(p2_cards))
            remaining_cards.pop(remaining_cards.index(p3_cards))

            t1 = threading.Thread(
                target=self.raw_p1_range_query,
                args=(unknown_range, p2_cards, p3_cards, "".join(remaining_cards), board, main_dictionary, online)
            )
            threads.append(t1)

            t1.start()

        for thread in threads:
            thread.join()

        del threads

        main_dictionary["metadata"] = "Using range - {}".format(
            self.unknown_range_1)
        return main_dictionary

    def run_function_with_existing_cards(self, func_name, unknown_range=""):
        """
        Takes in a function name and runs it with the existing cards
        Args:
            func_name: Name of the function to run
            unknown_range: Range of unknown cards
        Returns the dictionary of all the cards and their equity running according to their function
        """
        if func_name == "ev":
            return self.run_ev_query_for_multiple_cards(*self.cards, board="".join(self.board), online=False)
        elif func_name == "best":
            if unknown_range == "":
                unknown_range = self.unknown_range_1
            return self.run_best_hand_query_for_multiple_hands(*self.cards, board="".join(self.board), unknown_range=unknown_range, online=False)
        elif func_name == "p1_range":
            if unknown_range == "":
                unknown_range = self.unknown_range_1
            return self.run_p1_range_query_for_multiple_hands(*self.cards, board="".join(self.board), unknown_range=unknown_range, online=False)

    def threaded_run_function_with_existing_cards(self, data, func_name, unknown_range=""):
        """
        Threaded version of the run_function_with_existing_cards
        Args:
            data: Dictionary to append the results to
            func_name: Name of the function to run
            unknown_range: Range of unknown cards
        Appends the results to the dictionary
        Returns None
        """
        if func_name == "ev":
            data["ev"] = self.run_ev_query_for_multiple_cards(
                *self.cards, board="".join(self.board), online=False)  # self.online)
        elif func_name == "best":
            if unknown_range == "":
                unknown_range = self.unknown_range_1
            data["best"] = self.run_best_hand_query_for_multiple_hands(
                *self.cards, board="".join(self.board), unknown_range=unknown_range, online=False)  # self.online)
        elif func_name == "p1_range":
            if unknown_range == "":
                unknown_range = self.unknown_range_1
            data["p1_range"] = self.run_p1_range_query_for_multiple_hands(
                *self.cards, board="".join(self.board), unknown_range=unknown_range, online=False)  # self.online)

    def run_flop_weakness_5card(self, *args, **kwargs):
        """
        Runs the flop category query for the opponent
        Run this function only if its a 5card hand
        """
        # board = kwargs.get("board", "")
        if not len(self.board):
            board = ""
        else:
            board = "".join(list((self.board)))
        deadcards = (''.join(self.cards)).replace(" ", "")
        player_range = self.unknown_range_1
        # print(player_range)

        query = """select count(exactHandType(PLAYER_1,river,highcard)) /* How often PLAYER_1 5-card hand type is nothing by the river */ as COUNT3, count(exactHandType(PLAYER_1,river,pair)) /* How often PLAYER_1 5-card hand type is one pair by the river */ as COUNT4, count(exactHandType(PLAYER_1,river,twopair)) /* How often PLAYER_1 5-card hand type is two pair by the river */ as COUNT5, count(exactHandType(PLAYER_1,river,trips)) /* How often PLAYER_1 5-card hand type is trips by the river */ as COUNT6, count(exactHandType(PLAYER_1,river,straight)) /* How often PLAYER_1 5-card hand type is a straight by the river */ as COUNT7, count(exactHandType(PLAYER_1,river,flush)) /* How often PLAYER_1 5-card hand type is a flush by the river */ as COUNT8, count(exactHandType(PLAYER_1,river,fullhouse)) /* How often PLAYER_1 5-card hand type is a full house by the river */ as COUNT9, count(exactHandType(PLAYER_1,river,quads)) /* How often PLAYER_1 5-card hand type is quads by the river */ as COUNT10, count(exactHandType(PLAYER_1,river,straightflush)) /* How often PLAYER_1 5-card hand type is a straight flush by the river */ as COUNT11  from game='omahahi5', syntax='Generic',      board='{board}',      dead='{deadcards}',      PLAYER_1='{player_range}' """.format(
            deadcards=deadcards,
            board=board,
            player_range=player_range
        )
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )
        # print(query_string)
        resp = subprocess.check_output(query_string).decode('utf8')
        # print(resp)
        lines = resp.split("\n")
        # print(lines)
        data = {}
        for line in lines:
            if line.startswith("COUNT"):
                line = line.split('%')[0]
                line = line.replace(" ", "")
                key, value = line.split("=")
                if key == 'COUNT3':
                    key = "nothing"
                elif key == 'COUNT4':
                    key = "one_pair"
                elif key == 'COUNT5':
                    key = "two_pair"
                elif key == 'COUNT6':
                    key = "trips"
                elif key == 'COUNT7':
                    key = "straight"
                elif key == 'COUNT8':
                    key = "flush"
                elif key == 'COUNT9':
                    key = "full_house"
                elif key == 'COUNT10':
                    key = "quads"
                elif key == 'COUNT11':
                    key = "straight_flush"
                if float(value) > 2:
                   data[key] = float(value)
        # print(data)
        # create a pie chart of the data
        # labels = list(data.keys())
        # sizes = list(data.values())
        # fig1, ax1 = plt.subplots()
        # ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        #         shadow=True, startangle=90)
        # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # plt.show()
        weakness = sum(data.values()) - sum(list(data.values())[-2:])
        # print(weakness)
        return weakness


    def postflop_info(self, board, player_range, deadcards):
        """
        Runs the postflop category query for the opponent
        Run this function only if it's a 5card hand
        """
        # print("Inputs:", board, player_range, deadcards)

        if type(board) == list:
            board = "".join(board)


        if type(deadcards) == list:
            dc_temp = []
            for item in deadcards:
                if type(item) == list:
                    item = "".join(item)
                dc_temp.append(item)
            deadcards = dc_temp
            deadcards = "".join(deadcards)

        if type(player_range) == list:
            player_range = "".join(player_range)

        # print("Board: ", board)
        # print("Deadcards: ", deadcards)
        # print("Player Range: ", player_range)

        query = """select count(nutHi(PLAYER_1,river)) as COUNT1, count(minHandType(PLAYER_1,river,fullhouse)) as COUNT4, count(exactHandType(PLAYER_1,river,flush)) as COUNT5, count(exactHandType(PLAYER_1,river,straight)) as COUNT6 from game='omahahi5', syntax='Generic', board='{board}', dead='{deadcards}', PLAYER_1='{player_range}' """.format(
            deadcards=deadcards,
            board=board,
            player_range=player_range
        )
        query_string = self.command.format(
            folderpath=self.path_to_jar_folder,
            max_iterations=self.num_iterations,
            max_seconds=self.max_time,
            query=query
        )
        # print(query_string)
        resp = subprocess.check_output(query_string).decode('utf8')
        # print(resp)
        lines = resp.split("\n")
        # print(lines)
        data = {}
        for line in lines:
            if line.startswith("COUNT"):
                line = line.split('%')[0]
                line = line.replace(" ", "")
                key, value = line.split("=")
                if key == 'COUNT1':
                    key = "nut_hi"
                elif key == 'COUNT4':
                    key = "full_house_or_better"
                elif key == 'COUNT5':
                    key = "flush"
                elif key == 'COUNT6':
                    key = "straight"
                if float(value) > 2:
                    data[key] = float(value)
        # print(data)
        return data


class CardsValidator(object):

    def __init__(self):
        pass

    def validate_cards(self, *args):
        cards = list(args)
        unique_cards = list()
        for card in cards:

            for c in (card[i:i+2] for i in range(0, len(card), 2)):
                if c.lower() in unique_cards:
                    print(c, card)
                unique_cards.append(c.lower())

        if len(unique_cards) == len(set(unique_cards)):
            return True

        return False

if __name__ == '__main__':
    logger = logging.getLogger("test_logger")
    pwe = PlayerWithEquity(logger)
    val = CardsValidator()
    # generate random cards
    for i in range(1):
        pwe.reset_variables()
        deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
        deck = ["".join(card) for card in deck]
        random.shuffle(deck)
        n = 4
        num_cards_per_hand = 5
        cards = list()
        for i in range(n):
            cards.append(''.join(deck[i*num_cards_per_hand:(i+1)*num_cards_per_hand]))
        print(cards)
        board = deck[n*num_cards_per_hand:(n*num_cards_per_hand+3)]
        board = "".join(board)
        print(board)
        # print(board)
        dc = ''.join(cards[1:])
        print(dc)
        # print(pwe.postflop_info(board,cards[0],cards[1:]))
        pwe.add_cards(*cards)
        # pwe.unknown_range_1 = "25%"
        t1 = time.time()
        result1 = pwe.run_function_with_existing_cards("best")
        results_pair = pwe.run_function_with_existing_cards("p1_range")
        for pair, result in results_pair.items():
            try:
                equalized_result = 33 - result[pwe.default_range]
            except Exception as e:
                continue
        t1 = time.time()
        results_pair1 = pwe.pairwise_with_dc(player_range="25%", p2_cards=cards[0], p3_cards=cards[1], deadcards="".join(cards[2:]), board="")

        print(results_pair1)



        # sorted_results = dict(sorted(result.items()))
        # for key, value in sorted_results.items():
        #     print(key, ":", value)
        # t2 = time.time()
        # print('Time Taken to run best: ', t2-t1)
        # t1 = time.time()
        # print('Checking for only one hand query')
        # result = pwe.equity_with_dc(cards[0], "".join(cards[1:]), "25%", board="", online=False)
        # for key, value in result.items():
        #     print(key, ":", value)
        # t2 = time.time()
        # print('Time Taken to run one: ', t2-t1)

        # cards = ['As5d3h2d8h', '2c9cAd6dKs', '7c4dAh8dKc', '6sKd6c8cTc', '9h6hQd4cKh']
        # flop = deck[n*num_cards_per_hand:(n*num_cards_per_hand+3)]
        # print(val.validate_cards(*cards))
        # exit()
        # pprint(pwe.run_p1_range_query_for_multiple_hands(
        #     "9ctd2c2h7c", "ts6s7s8hqh", "khjsjc6h5d", "th8c4c3d2d", "kdqc8s6c3h", board="acjd4s4d"))
        # pwe.add_cards(*cards)
        # pwe.add_to_board("acjd4d")

        # if random.randint(0, 1):
        #     pwe.add_to_board(*flop)

        # print(pwe.run_function_with_existing_cards("best"))
        # print(pwe.run_function_with_existing_cards("p1_range"))

        # pwe.set_q1_range("25%")
        # pwe.run_function_with_existing_cards("ev","100%",)
        # pwe.run_flop_weakness_5card()

    # pprint(pwe.run_best_hand_query_for_multiple_hands(
    #     "9ctd2c2h7c", "ts6s7s8hqh", "khjsjc6h5d", "th8c4c3d2d", "kdqc8s6c3h",
    #     # board="acjd4s4d",
    #     unknown_range='30%'))
    # pwe.set_mode("all")
    # t1 = time.time()
    # pwe.set_to_online_mode()
    # print(pwe.run_function_with_existing_cards("p1_range"))
    # t2 = time.time()
    # print("Time taken online", t2-t1)
    # print(pwe.run_query_online("""
    # select /* Start equity stats */
    #     avg(riverEquity(PLAYER_1)) as PLAYER_1_equity1,
    #         count(winsHi(PLAYER_1)) as PLAYER_1_winsHi1,
    #         count(tiesHi(PLAYER_1)) as PLAYER_1_tiesHi1,
    #         avg(riverEquity(PLAYER_2)) as PLAYER_2_equity1,
    #         count(winsHi(PLAYER_2)) as PLAYER_2_winsHi1,
    #         count(tiesHi(PLAYER_2)) as PLAYER_2_tiesHi1
    #     /* End equity stats */
    #     from game='omahahi5', syntax='Generic',
    #         board='',
    #         dead='KhJsJc6h5dTs6s7s8hQh9cTd2c2h7cKdQc8s6c3h',
    #         PLAYER_1='Th8c4c3d2d',
    #         PLAYER_2='100%'
    # """))
    # print(pwe.cards)
