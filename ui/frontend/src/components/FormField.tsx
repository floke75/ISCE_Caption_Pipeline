import { PropsWithChildren } from "react";
import clsx from "clsx";

type Props = PropsWithChildren<{
  label: string;
  description?: string;
  error?: string;
  inline?: boolean;
}>;

export function FormField({ label, description, error, inline, children }: Props) {
  return (
    <label
      className={clsx(
        "flex w-full",
        inline ? "items-center space-x-3" : "flex-col space-y-2"
      )}
    >
      <span className="text-sm font-medium text-slate-200">
        {label}
        {description && <span className="ml-2 text-xs text-slate-400">{description}</span>}
      </span>
      <div className={clsx("w-full", inline ? "max-w-sm" : "w-full space-y-1")}>{children}</div>
      {error && <span className="text-xs text-red-400">{error}</span>}
    </label>
  );
}
