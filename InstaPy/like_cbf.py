from instapy import InstaPy
import configparser

config = configparser.ConfigParser()
config.read("C:\Windows\System32\drivers\etc\config.txt")
name = config.get("cbf","cbf_name")
key = config.get("cbf","cbf_key")

session = InstaPy(username=name, password=key)
session.login()

session.set_smart_hashtags(['travel', 'blog' ], limit=50, sort='top', log_tags=True)
session.like_by_tags(amount=100, use_smart_hashtags=True)
session.set_user_interact(amount=3, randomize=True, percentage=100, media='Photo')
session.like_by_feed(amount=200, randomize=True, unfollow=True, interact=True)

session.end()

