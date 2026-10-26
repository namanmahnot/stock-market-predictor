/** Account panel: guest badge, sign in, register, sign out. */
import { useState } from "react";

import { describeError } from "../lib/api";
import { useSession } from "../lib/session";
import { Badge, buttonClass, secondaryButtonClass } from "./ui";

export function AccountMenu() {
  const { user, isGuest, signIn, signUp, signOut } = useSession();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") await signIn(email, password);
      else await signUp(email, password);
      setOpen(false);
      setEmail("");
      setPassword("");
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-xs font-medium text-ink hover:border-accent"
      >
        <span aria-hidden="true">👤</span>
        <span className="hidden max-w-[16ch] truncate sm:inline">
          {user ? (isGuest ? "Guest session" : user.email) : "No session"}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 z-40 mt-2 w-80 rounded-xl border border-line bg-surface-raised p-4 shadow-xl">
          <div className="mb-3 flex items-center justify-between gap-2">
            <p className="text-sm font-semibold text-ink">Account</p>
            {isGuest && <Badge tone="warning">guest</Badge>}
          </div>
          <p className="mb-3 text-xs text-ink-secondary">
            {user
              ? isGuest
                ? "A local guest account keeps your watchlist and saved runs on this browser. Register to keep them under your own email."
                : `Signed in as ${user.email}.`
              : "No session is active, so saving is disabled."}
          </p>

          {user && !isGuest ? (
            <button type="button" className={secondaryButtonClass} onClick={signOut}>
              Sign out
            </button>
          ) : (
            <form onSubmit={submit} className="space-y-2">
              <div className="flex gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => setMode("login")}
                  className={`rounded-md px-2 py-1 ${mode === "login" ? "bg-accent-soft text-accent" : "text-ink-secondary"}`}
                >
                  Sign in
                </button>
                <button
                  type="button"
                  onClick={() => setMode("register")}
                  className={`rounded-md px-2 py-1 ${mode === "register" ? "bg-accent-soft text-accent" : "text-ink-secondary"}`}
                >
                  Register
                </button>
              </div>
              <label className="block text-xs font-medium text-ink-secondary">
                Email
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
                />
              </label>
              <label className="block text-xs font-medium text-ink-secondary">
                Password
                <input
                  type="password"
                  required
                  minLength={8}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
                />
                <span className="mt-1 block font-normal text-ink-muted">At least 8 characters.</span>
              </label>
              {error && <p className="text-xs text-[var(--color-critical)]">{error}</p>}
              <button type="submit" className={buttonClass} disabled={busy}>
                {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
              </button>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
