'use client';

import { memo, useEffect, useId, useRef } from 'react';

const EAST_BEACH = 'M1498-70C1406 79 1510 208 1473 355S1452 499 1560 603 1598 813 1715 883L1730-90Z';
const WEST_BEACH = 'M-90 877C-11 910-13 1012 95 1044S189 1054 237 1146H-90Z';

/** A deterministic estuary illustration: shallow water, sand banks and tidal contours. */
export const WaterScene = memo(function WaterScene() {
  const ref = useRef<SVGSVGElement>(null);
  const id = useId().replace(/:/g, '');

  useEffect(() => {
    const scene = ref.current;
    if (!scene) return;
    let visible = true;
    const update = () => {
      scene.style.setProperty('--scene-play-state', document.hidden || !visible ? 'paused' : 'running');
    };
    document.addEventListener('visibilitychange', update);
    const observer = typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        update();
      })
      : null;
    observer?.observe(scene);
    update();
    return () => {
      document.removeEventListener('visibilitychange', update);
      observer?.disconnect();
    };
  }, []);

  return (
    <svg
      ref={ref}
      aria-hidden="true"
      focusable="false"
      data-water-scene
      className="water-scene"
      preserveAspectRatio="xMidYMid slice"
      viewBox="0 0 1600 1100"
    >
      <defs>
        <linearGradient id={`${id}-sea`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#377f75" />
          <stop offset=".44" stopColor="#175a59" />
          <stop offset="1" stopColor="#062f35" />
        </linearGradient>
        <radialGradient id={`${id}-shallows`} cx=".1" cy=".02" r=".8">
          <stop stopColor="#b2e6ce" stopOpacity=".38" />
          <stop offset=".4" stopColor="#65bdab" stopOpacity=".19" />
          <stop offset="1" stopColor="#327f7d" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`${id}-sand`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#f9e8c7" />
          <stop offset=".38" stopColor="#e7cb99" />
          <stop offset=".7" stopColor="#d9b881" />
          <stop offset="1" stopColor="#f0d6a9" />
        </linearGradient>
        <pattern id={`${id}-grain`} width="29" height="23" patternUnits="userSpaceOnUse">
          <path d="M3 4h1.1M17 2h.6M24 11h1.2M11 18h.7M4 21h.9M19 17h.8" stroke="#876b43" strokeWidth=".65" opacity=".4" />
          <path d="M8 8h1.3M22 4h.7M14 14h1M2 13h.8M26 21h1.1" stroke="#fff5dc" strokeWidth=".8" opacity=".7" />
        </pattern>
        <clipPath id={`${id}-beaches`}>
          <path d={EAST_BEACH} />
          <path d={WEST_BEACH} />
        </clipPath>
        <linearGradient id={`${id}-shade`} x1="0" y1="0" x2="0" y2="1">
          <stop stopColor="#00282c" stopOpacity="0" />
          <stop offset="1" stopColor="#00282c" stopOpacity=".56" />
        </linearGradient>
      </defs>
      <path d="M0 0H1600V1100H0Z" fill={`url(#${id}-sea)`} />
      <path d="M0 0H1600V1100H0Z" fill={`url(#${id}-shallows)`} />
      <g className="water-drift water-contours" fill="none" stroke="#a0d4c0" strokeWidth="1">
        {Array.from({ length: 21 }, (_, i) => (
          <path key={i} data-water-contour d={`M ${-280 + i * 17} ${30 + i * 42} C ${110 + i * 23} ${-110 + i * 44}, ${360 + i * 5} ${95 + i * 37}, ${530 + i * 11} ${250 + i * 31} S ${1160 + i * 6} ${310 + i * 29}, ${1800 + i * 8} ${-45 + i * 51}`} opacity={i % 3 === 0 ? '.13' : '.06'} />
        ))}
      </g>
      <g className="water-drift water-caustics" fill="none" stroke="#e0f2d9" strokeWidth="1.2" opacity=".13">
        <path d="M-130 125C10 51 87 86 117 142S274 214 389 158 574 86 638 177 793 230 843 169M-78 172C71 125 115 190 205 250S354 213 397 269 532 295 610 256" />
        <path d="M-75 344C73 294 123 333 143 397S298 458 372 390 535 399 561 476M28 602C101 489 194 557 241 534S376 512 412 584 532 689 630 626" />
        <path d="M534 78C648 42 704 82 757 100S866 72 931 126 1049 219 1158 150M929 372C1044 318 1123 353 1167 426S1298 436 1371 459 1496 579 1649 526" />
        <path d="M1146 744C1266 665 1327 741 1340 788S1496 882 1699 796M-128 820C14 739 93 824 168 819S302 733 407 825 533 932 645 882" />
      </g>
      <g data-shoreline>
        <path d="M1436-70C1329 80 1459 218 1419 360S1381 524 1499 625 1537 855 1685 943L1730-90Z" fill="#8bb9a2" opacity=".18" />
        <path d="M1468-70C1367 80 1483 213 1447 357S1414 510 1530 614 1566 852 1698 906L1730-90Z" fill="#aac9b0" opacity=".24" />
        <path d={EAST_BEACH} fill={`url(#${id}-sand)`} />
        <path d="M1498-70C1406 79 1510 208 1473 355S1452 499 1560 603 1598 813 1715 883" fill="none" stroke="#bda477" strokeWidth="10" opacity=".7" />
        <path d="M1493-70C1401 79 1505 208 1468 355S1447 499 1555 603 1593 813 1710 883" fill="none" stroke="#f5f5df" strokeWidth="2" opacity=".8" />
        <path d="M-90 803C55 829 11 958 124 986S256 1001 315 1146H-90Z" fill="#86b4a1" opacity=".17" />
        <path d="M-90 842C21 867-1 986 111 1014S226 1028 276 1146H-90Z" fill="#bfd0ad" opacity=".27" />
        <path d={WEST_BEACH} fill={`url(#${id}-sand)`} />
        <g clipPath={`url(#${id}-beaches)`}>
          <path d="M0 0H1730V1150H-90V0Z" fill={`url(#${id}-grain)`} />
          <g fill="none" stroke="#fff3d6" strokeWidth="1.5" opacity=".32">
            <path d="M1516-50C1444 86 1549 191 1514 360S1503 532 1588 620 1642 831 1740 866" />
            <path d="M1544-50C1477 97 1575 191 1541 363S1530 540 1615 627 1669 838 1767 873" />
            <path d="M1575-50C1508 97 1606 191 1572 363S1561 540 1646 627 1700 838 1798 873" />
            <path d="M-70 937C-2 966 4 1036 87 1076S153 1081 183 1159" />
            <path d="M-85 960C-17 989-11 1059 72 1099S138 1104 168 1182" />
          </g>
        </g>
      </g>
      <path d="M0 0H1600V1100H0Z" fill={`url(#${id}-shade)`} />
    </svg>
  );
});
