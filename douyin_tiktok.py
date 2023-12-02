import asyncio
from douyin_tiktok_scraper.scraper import Scraper
import urllib.request

api = Scraper()

async def hybrid_parsing(url: str) -> dict:
    # Hybrid parsing(Douyin/TikTok URL)
    result = await api.hybrid_parsing(url)
    print(f"The hybrid parsing result:\n {result}")
    return result


link = ['https://www.tiktok.com/@kendrick.bbq/video/7304708149573963050?is_from_webapp=1&sender_device=pc',
        'https://www.tiktok.com/@kendrick.bbq/video/6988142058477292805?is_from_webapp=1&sender_device=pc&web_id=7306892682533357057',
        'https://www.tiktok.com/@forzahorizon5_xpert_/video/7275597383008914695?is_from_webapp=1&sender_device=pc',
        'https://www.tiktok.com/@benejaamin/video/7287275949085707552?is_from_webapp=1&sender_device=pc']

video = []
for l in link:
    x = asyncio.run(hybrid_parsing(url= l))
    video.append(x['video_data']['nwm_video_url'])

c= 0
for i in video:
    c += 1
    urllib.request.urlretrieve(i, f'video_name_{c}.mp4')
print('d')