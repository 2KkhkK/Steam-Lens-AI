"""캐시 기반 요청 제한과 관리자 로그인 실패 제한.

외부 패키지(django-ratelimit, django-axes) 대신 직접 구현한 이유
  - 필요한 기능이 "고정 창(fixed window) 카운터" 하나뿐이라 코드가 짧다.
  - 동작을 테스트로 완전히 검증할 수 있다.
  - django-axes는 별도 테이블과 마이그레이션이 필요하다.

한계(알고 쓸 것)
  - 카운터는 Django 캐시에 저장된다. 기본 LocMemCache는 프로세스 로컬이라
    워커가 N개면 실제 한도가 N배가 된다. 운영에서는 REDIS_URL을 지정한다.
  - 고정 창 방식이라 창 경계에서 최대 2배까지 몰릴 수 있다. 이 서비스의
    목적(남용 차단)에는 충분하다.
"""

import logging
import re
import time

from django.conf import settings
from django.contrib import admin
from django.core.cache import cache
from django.http import HttpResponse

from .utils import cache_key

logger = logging.getLogger(__name__)

_RATE_RE = re.compile(r'^\s*(\d+)\s*/\s*(\d*)\s*([smhd])\s*$')
_UNIT_SECONDS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}


def parse_rate(rate):
    """'30/m', '5/15m' 같은 문자열을 (허용 횟수, 창 길이 초)로 변환한다."""
    match = _RATE_RE.match(str(rate or ''))
    if not match:
        raise ValueError(f'잘못된 rate 형식입니다: {rate!r} (예: "30/m", "5/15m")')
    count, multiplier, unit = match.groups()
    window = int(multiplier or 1) * _UNIT_SECONDS[unit]
    return int(count), window


def _conf(name, default=None):
    return getattr(settings, 'RATE_LIMITS', {}).get(name, default)


def client_ip(request):
    """요청자의 IP.

    X-Forwarded-For는 클라이언트가 마음대로 보낼 수 있는 헤더다. 프록시 뒤에
    있다고 명시했을 때만 믿고, 그때도 '우리 프록시가 마지막에 덧붙인' 맨 오른쪽
    값을 쓴다. 맨 왼쪽 값을 쓰면 공격자가 헤더를 위조해 제한을 피할 수 있다.
    """
    if getattr(settings, 'BEHIND_PROXY', False):
        forwarded = request.META.get('HTTP_X_FORWARDED_FOR', '')
        hops = [h.strip() for h in forwarded.split(',') if h.strip()]
        if hops:
            return hops[-1]
    return request.META.get('REMOTE_ADDR', '') or 'unknown'


def hit(group, ident, rate):
    """카운터를 1 올리고 한도를 넘었는지 반환한다. (초과 여부, 현재 횟수)"""
    limit, window = parse_rate(rate)
    bucket = int(time.time() // window)
    key = cache_key(f'rl_{group}', f'{ident}:{bucket}')

    # add는 키가 없을 때만 성공한다. 경쟁 상황에서도 초기화가 한 번만 일어난다.
    cache.add(key, 0, timeout=window + 5)
    try:
        count = cache.incr(key)
    except ValueError:
        # add와 incr 사이에 만료된 극단적인 경우
        cache.set(key, 1, timeout=window + 5)
        count = 1
    return count > limit, count


def is_rate_limited(request, group, ident=None):
    """이 요청이 group의 한도를 넘었는가. 비활성화돼 있으면 항상 False."""
    if not _conf('ENABLED', True):
        return False
    rate = _conf(group)
    if not rate:
        return False
    ident = ident or client_ip(request)
    limited, count = hit(group.lower(), ident, rate)
    if limited:
        logger.warning('요청 제한 초과 [%s] ident=%s count=%d rate=%s', group, ident, count, rate)
    return limited


# -----------------------------------------------------------------------------
# 관리자 로그인 실패 제한
# -----------------------------------------------------------------------------
def _failure_keys(request):
    username = str(request.POST.get('username', '')).strip().lower()[:150]
    keys = [cache_key('admin_fail_ip', client_ip(request))]
    if username:
        keys.append(cache_key('admin_fail_user', username))
    return keys


def _failure_count(key):
    return cache.get(key, 0)


def _record_failure(key, window):
    cache.add(key, 0, timeout=window)
    try:
        cache.incr(key)
    except ValueError:
        cache.set(key, 1, timeout=window)


def throttled_admin_login(request, extra_context=None):
    """admin.site.login을 감싸 로그인 '실패'를 센다.

    IP 기준과 사용자명 기준을 따로 센다.
      - IP 기준만 있으면 여러 IP로 한 계정을 공격하는 분산 대입을 못 막는다.
      - 사용자명 기준만 있으면 한 IP가 여러 계정을 돌아가며 시도하는 것을 못 막는다.
    트레이드오프: 사용자명 기준 잠금은 공격자가 관리자 계정을 일부러 잠글 수 있다는
    뜻이기도 하다. 잠금은 창 길이(기본 15분) 뒤 자동으로 풀린다.

    차단 중에는 비밀번호가 맞아도 거부한다. 그래야 잠금 중 대입이 무의미해진다.
    """
    if request.method != 'POST' or not _conf('ENABLED', True):
        return admin.site.login(request, extra_context)

    limit, window = parse_rate(_conf('ADMIN_LOGIN_FAILURES', '5/15m'))
    keys = _failure_keys(request)

    if any(_failure_count(k) >= limit for k in keys):
        logger.warning('관리자 로그인 차단: ip=%s', client_ip(request))
        return HttpResponse(
            '로그인 시도가 너무 많습니다. 잠시 후 다시 시도해 주세요.',
            status=429,
            content_type='text/plain; charset=utf-8',
        )

    response = admin.site.login(request, extra_context)

    # 성공하면 admin.site.login 안에서 request.user가 로그인 사용자로 바뀐다.
    if not (request.user.is_authenticated and request.user.is_staff):
        for key in keys:
            _record_failure(key, window)
        logger.info('관리자 로그인 실패: ip=%s', client_ip(request))

    return response
