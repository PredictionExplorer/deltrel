import { GameApp } from '@/components/GameApp';
import { hasSelfPlayAccess } from '@/lib/self-play-access';

export default async function Home({ searchParams }: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const allowSelfPlay = hasSelfPlayAccess(query.selfplay, process.env.DELTREL_SELF_PLAY_SECRET);
  return <GameApp allowSelfPlay={allowSelfPlay} />;
}
