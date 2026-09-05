import { memo, useEffect, useRef } from 'react';
import { mountStage, unmountStage } from './controller';

export const TheaterStage = memo(function TheaterStage() {
  'use no memo';
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = hostRef.current;
    if (node === null) return;
    mountStage(node);
    return () => {
      unmountStage();
    };
  }, [mountStage, unmountStage]);

  return <div ref={hostRef} className="absolute inset-0" />;
});
