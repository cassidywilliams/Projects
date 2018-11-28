from instapy import InstaPy

insta_username = 'peopleofhamburg'
insta_password = 'number03'

# if you want to run this script on a server,
# simply add nogui=True to the InstaPy() constructor
session = InstaPy(username=insta_username, password=insta_password)
session.login()

# set up all the settings
session.set_upper_follower_count(limit=2500)
session.set_do_comment(True, percentage=40)
session.set_comments(['Amazing! @{}!','Nice photo @{} :D', 'Great photo! @{}!', 'Cool photo!'])
session.set_do_follow(enabled=True, percentage=70, times=1)
session.set_dont_include(['offeneblende'])

# do the actual liking
session.set_user_interact(amount=3, randomize=True, percentage=100, media='Photo')
session.like_by_tags(['boating', 'sailaway', 'boatlife', 'yachtparty', 'sailboat', 'sailinglife', 'yachtlife', 'luxuryyacht', 'yachting', 'superyachts', 'yachtclub', 'motoryacht', 'superyacht', 'yachtdesign', 'charter', 'megayachts', 'billionairetoys', 'yachtinglifestyle', 'megayacht', 'yachtworld', 'sailingboat', 'yachtlifestyle', 'diewocheaufinstagram', 'ig_hamburg', 'welovehh', 'igershh', 'hamburgmeineperle', 'typischhamburch'], amount=100)
session.follow_user_following(['heute_in_hamburg', 'hamburgbestof', 'hamburgermorgenpost'], amount=25, randomize=False)
session.set_dont_unfollow_active_users(enabled=True, posts=5)
session.unfollow_users(amount=100, onlyInstapyFollowed = True, onlyInstapyMethod = 'FIFO', sleep_delay=60 )
session.like_by_locations(['226981041', '213110159', '458552132'], amount=100)

# end the bot session
session.end()

