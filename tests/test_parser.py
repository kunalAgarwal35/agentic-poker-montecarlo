from pql.parser.ast import Query, SelectItem, Call, Ident
from pql.parser.parse import parse


def test_ast_nodes_construct():
    item = SelectItem(
        expr=Call(name="avg", args=[Call(name="riverEquity", args=[Ident("PLAYER_1")])]),
        alias="p1_eq",
    )
    q = Query(select=[item], params={"game": "omahahi5"})
    assert q.select[0].alias == "p1_eq"
    assert q.select[0].expr.name == "avg"
    assert q.select[0].expr.args[0].name == "riverEquity"
    assert q.select[0].expr.args[0].args[0].name == "PLAYER_1"
    assert q.params["game"] == "omahahi5"


def test_parse_two_column_query():
    text = (
        "select avg(riverEquity(PLAYER_1)) as p1_eq, "
        "count(winsHi(PLAYER_1)) as p1_wins "
        "from game='omahahi5', syntax='Generic', board='2c3c4c', "
        "dead='', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    q = parse(text)
    assert len(q.select) == 2
    assert q.select[0].alias == "p1_eq"
    assert q.select[0].expr.name == "avg"
    assert q.select[0].expr.args[0].name == "riverEquity"
    assert q.select[0].expr.args[0].args[0].name == "PLAYER_1"
    assert q.select[1].expr.name == "count"
    assert q.select[1].alias == "p1_wins"
    assert q.select[1].expr.args[0].name == "winsHi"
    assert q.params["game"] == "omahahi5"
    assert q.params["board"] == "2c3c4c"
    assert q.params["dead"] == ""
    assert q.params["player_1"] == "AsAhKsKhQs"


def test_parse_ignores_block_comments():
    text = (
        "select /* equity */ avg(riverEquity(PLAYER_1)) as p1 "
        "from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    q = parse(text)
    assert q.select[0].alias == "p1"


def test_parse_unaliased_item_has_none_alias():
    text = "select avg(riverEquity(PLAYER_1)) from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    q = parse(text)
    assert q.select[0].alias is None


def test_parse_comment_with_star_inside():
    text = (
        "select /* a*b */ avg(riverEquity(PLAYER_1)) as p1 "
        "from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    q = parse(text)
    assert q.select[0].alias == "p1"


def test_parse_empty_comment():
    text = (
        "select /**/ avg(riverEquity(PLAYER_1)) as p1 "
        "from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    q = parse(text)
    assert q.select[0].alias == "p1"
