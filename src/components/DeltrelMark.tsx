import type { SVGProps } from 'react';

/** Three tributaries joining a single watercourse. */
export function DeltrelMark(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" {...props}>
      <path d="M9 7C9 18 24 15 24 27V42M24 6V42M39 7C39 18 24 15 24 27" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M7 29C13 25 17 31 24 31C31 31 35 25 41 29" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".55" />
    </svg>
  );
}
