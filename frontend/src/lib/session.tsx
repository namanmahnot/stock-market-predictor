/**
 * Session handling.
 *
 * Watchlists and saved prediction runs belong to a user, so the dashboard needs
 * an account. To keep a lab demonstration one click away, the app creates a
 * local guest account on first load; the account menu lets a user register or
 * sign in properly, which keeps their saved work under their own email.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { AuthApi, readToken, storeToken } from "./api";
import type { User } from "./types";

const GUEST_FLAG = "spps.guest";

interface SessionContextValue {
  user: User | null;
  isGuest: boolean;
  ready: boolean;
  error: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const SessionContext = createContext<SessionContextValue | null>(null);

function randomGuest(): { email: string; password: string } {
  const suffix = crypto.randomUUID().slice(0, 12);
  return { email: `guest-${suffix}@local.demo`, password: `Guest-${crypto.randomUUID()}` };
}

function markGuest(isGuest: boolean): void {
  try {
    if (isGuest) localStorage.setItem(GUEST_FLAG, "1");
    else localStorage.removeItem(GUEST_FLAG);
  } catch {
    // Non-fatal: the badge in the account menu is the only thing affected.
  }
}

function wasGuest(): boolean {
  try {
    return localStorage.getItem(GUEST_FLAG) === "1";
  } catch {
    return false;
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isGuest, setIsGuest] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function start(): Promise<void> {
      try {
        if (readToken()) {
          const existing = await AuthApi.me();
          if (!cancelled) {
            setUser(existing);
            setIsGuest(wasGuest());
          }
          return;
        }
        const guest = randomGuest();
        const session = await AuthApi.register(guest.email, guest.password);
        storeToken(session.access_token);
        markGuest(true);
        if (!cancelled) {
          setUser(session.user);
          setIsGuest(true);
        }
      } catch {
        // An expired or rejected token should not trap the user on a blank page.
        storeToken(null);
        if (!cancelled) {
          setUser(null);
          setError("Could not start a session. The dashboard works without one, but saving is disabled.");
        }
      } finally {
        if (!cancelled) setReady(true);
      }
    }

    void start();
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const session = await AuthApi.login(email, password);
    storeToken(session.access_token);
    markGuest(false);
    setUser(session.user);
    setIsGuest(false);
    setError(null);
  }, []);

  const signUp = useCallback(async (email: string, password: string) => {
    const session = await AuthApi.register(email, password);
    storeToken(session.access_token);
    markGuest(false);
    setUser(session.user);
    setIsGuest(false);
    setError(null);
  }, []);

  const signOut = useCallback(() => {
    storeToken(null);
    markGuest(false);
    setUser(null);
    window.location.reload();
  }, []);

  const value = useMemo<SessionContextValue>(
    () => ({ user, isGuest, ready, error, signIn, signUp, signOut }),
    [user, isGuest, ready, error, signIn, signUp, signOut],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error("useSession must be used inside SessionProvider.");
  }
  return context;
}
