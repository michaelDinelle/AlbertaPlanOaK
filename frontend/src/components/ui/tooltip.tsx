import { Tooltip as TooltipPrimitive } from '@base-ui/react/tooltip';
import { clsx } from 'clsx';
import type { ComponentProps } from 'react';

export function TooltipProvider({ delay = 250, ...props }: ComponentProps<typeof TooltipPrimitive.Provider>) {
  return <TooltipPrimitive.Provider delay={delay} {...props} />;
}

export function Tooltip(props: ComponentProps<typeof TooltipPrimitive.Root>) {
  return <TooltipPrimitive.Root {...props} />;
}

export const TooltipTrigger = TooltipPrimitive.Trigger;

export interface TooltipContentProps extends ComponentProps<typeof TooltipPrimitive.Popup> {
  side?: ComponentProps<typeof TooltipPrimitive.Positioner>['side'];
  sideOffset?: number;
  align?: ComponentProps<typeof TooltipPrimitive.Positioner>['align'];
}

export function TooltipContent({
  className,
  side = 'top',
  sideOffset = 6,
  align = 'center',
  children,
  ...props
}: TooltipContentProps) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Positioner side={side} sideOffset={sideOffset} align={align} className="z-50">
        <TooltipPrimitive.Popup
          className={clsx(
            'glass-quiet rounded-sm px-2 py-1 font-sans text-2xs text-ink',
            'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0 transition-opacity duration-100',
            className,
          )}
          {...props}
        >
          {children}
        </TooltipPrimitive.Popup>
      </TooltipPrimitive.Positioner>
    </TooltipPrimitive.Portal>
  );
}
