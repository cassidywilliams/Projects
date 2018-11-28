from instapy import InstaPy
import configparser

config = configparser.ConfigParser()
config.read("C:\Windows\System32\drivers\etc\config.txt")
name = config.get("poh","poh_name")
key = config.get("poh","poh_key")

session = InstaPy(username=name, password=key)
session.login()

session.set_user_interact(amount=3, randomize=True, percentage=50, media='Photo')
session.set_do_like(enabled=True)
session.like_by_feed(amount=100, randomize=True, unfollow=True, interact=True)
session.unfollow_users(amount=10, onlyInstapyFollowed = True, onlyInstapyMethod = 'FIFO', sleep_delay=60 )
session.set_dont_unfollow_active_users(enabled=True, posts=5)

session.set_smart_hashtags(['sunset', 'city', 'hamburg', 'germany'], limit=25, sort='random', log_tags=True)
session.like_by_tags(amount=100, use_smart_hashtags=True)

session.set_user_interact(amount=3, randomize=True, percentage=50, media='Photo')
#session.set_do_follow(enabled=False, percentage=70)
session.set_do_like(enabled=True, percentage=70)
session.set_comments(["Beautiful!", "Great photo!", "Wonderful shot!"])
session.set_do_comment(enabled=True, percentage=80)
session.interact_user_followers(['aino_heuteinhamburg', 'hamburgahoi', 'hamburg_online', 'hamburgmorgenpost'], amount=100, randomize=True)


session.end()



