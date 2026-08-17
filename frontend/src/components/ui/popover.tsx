import { Popover as PopoverPrimitive } from '@base-ui/react/popover';
import { clsx } from 'clsx';
import type { ComponentProps } from 'react';

export function Popover(props: ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root {...props} />;
}

export const PopoverTrigger = PopoverPrimitive.Trigger;

export const PopoverClose = PopoverPrimitive.Close;

export interface PopoverContentProps extends ComponentProps<typeof PopoverPrimitive.Popup> {
  side?: ComponentProps<typeof PopoverPrimitive.Positioner>['side'];
  sideOffset?: number;
  align?: ComponentProps<typeof PopoverPrimitive.Positioner>['align'];
}

export function PopoverContent({
  className,
  side = 'bottom',
  sideOffset = 8,
  align = 'start',
  children,
  ...props
}: PopoverContentProps) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Positioner side={side} sideOffset={sideOffset} align={align} className="z-50">
        <PopoverPrimitive.Popup
          className={clsx(
            'glass rounded-md p-2 font-sans text-ink outline-none',
            'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0 transition-opacity duration-100',
            className,
          )}
          {...props}
        >
          {children}
        </PopoverPrimitive.Popup>
      </PopoverPrimitive.Positioner>
    </PopoverPrimitive.Portal>
  );
}

export function PopoverTitle({ className, ...props }: ComponentProps<typeof PopoverPrimitive.Title>) {
  return <PopoverPrimitive.Title className={clsx('text-sm font-medium', className)} {...props} />;
}

export function PopoverDescription({ className, ...props }: ComponentProps<typeof PopoverPrimitive.Description>) {
  return <PopoverPrimitive.Description className={clsx('text-2xs text-ink-dim', className)} {...props} />;
}
