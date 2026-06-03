from pql.parser.parse import parse
from pql.parser.ast import Call, Ident


def test_integer_arg_parses_as_ident_token():
    q = parse("select avg(minOutsToHandType(PLAYER_1, turn, flush, 9)) as x "
              "from game='holdem', PLAYER_1='KhQh', PLAYER_2='AsAd'")
    agg = q.select[0].expr                 # avg(...)
    value_call = agg.args[0]               # minOutsToHandType(...)
    assert isinstance(value_call, Call)
    assert value_call.name == "minOutsToHandType"
    # args: PLAYER_1, turn, flush, 9  -> all Idents; the int carries its digit string
    assert [a.name for a in value_call.args] == ["PLAYER_1", "turn", "flush", "9"]
    assert all(isinstance(a, Ident) for a in value_call.args)


def test_existing_queries_still_parse():
    q = parse("select count(minHandType(PLAYER_1, river, flush)) as m "
              "from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'")
    assert q.select[0].alias == "m"
