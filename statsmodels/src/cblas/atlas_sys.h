#ifndef ATLAS_SYS_H
   #define ATLAS_SYS_H
/*
 * This file contains routines to interact with the system (as in the C
 * `system' command), and related I/O
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

static char *NewStringCopy(char *old)
/*
 * RETURNS: newly allocates string containing copy of string old
 * NOTE: old is not modified.
 */
{
   char *new;
   if (!old)
      return(NULL);
   new = malloc(sizeof(char)*(strlen(old)+1));
   assert(new);
   strcpy(new, old);
   return(new);
}

static char *NewAppendedString0(char *old, char *app)
/*
 * RETURNS: string holding : old + app
 * NOTE: frees old string after copy
 */
{
   char *new;
   if (!old)
   {
      new = malloc(sizeof(char)*(strlen(app)+1));
      assert(new);
      strcpy(new, app);
   }
   else
   {
      new = malloc(sizeof(char)*(strlen(old) + strlen(app)+1));
      assert(new);
      strcpy(new, old);
      strcat(new, app);
      free(old);
   }
   return(new);
}

static char *NewAppendedString(char *old, char *app)
/*
 * RETURNS: string holding : old + " " + app
 * NOTE: frees old string after copy
 */

{
   char *new;
   if (!old)
   {
      new = malloc(sizeof(char)*(strlen(app)+1));
      assert(new);
      strcpy(new, app);
   }
   else
   {
      new = malloc(sizeof(char)*(strlen(old) + strlen(app)+2));
      assert(new);
      strcpy(new, old);
      strcat(new, " ");
      strcat(new, app);
      free(old);
   }
   return(new);
}

static char *NewAppendedStrings(char *old, char *app0, char *app1)
/*
 * RETURNS: string holding : old + " " + app0 + " " + app1
 * NOTE: frees old string after copy
 */

{
   char *new;
   int len;

   assert(app0 && app1);
   len = strlen(app0) + strlen(app1) + 2;
   if (!old)
   {
      new = malloc(sizeof(char)*len);
      assert(new);
      sprintf(new, "%s %s", app0, app1);
   }
   else
   {
      len += strlen(old) + 1;
      new = malloc(sizeof(char)*len);
      assert(new);
      sprintf(new, "%s %s %s", old, app0, app1);
      free(old);
   }
   return(new);
}

static char *ATL_fgets(char *sout, int *plen, FILE *fpin)
/*
 * This routine returns a pointer to a single line of of file fpin.
 * If the plen-length string sout is long enough to hold the file's line,
 * then sout will be the return value.  Otherwise sout will be freed and
 * a new string will be returned.
 * Upon EOF/error: sout is de-allocated, *len=0, & NULL is returned;
 * *len is the length of sout on input, and of the returned string on output.
 */
{
   int len = *plen;
   if (!sout || len < 1)
   {
      *plen = len = 128;
      sout = malloc(len*sizeof(char));
      assert(sout);
   }
/*
 * See if there is a line left in file
 */
   if (fgets(sout, len, fpin))
   {
      int i;
      for (i=0; sout[i]; i++);
      assert(i > 0);
      if (sout[i-1] == '\n')    /* if this is complete line */
         return(sout);          /* we are done, return it */
/*
 *    Continue doubling string length until we can fit the whole string
 */
      while (sout[i-1] != '\n')
      {
         char *sp;
         int len0 = len;

         *plen = (len += len);
         sp = malloc(len*sizeof(char));
         assert(sp);
         strcpy(sp, sout);
         free(sout);
         sout = sp;
         sp += i;
         if (!fgets(sp, len0, fpin))
            return(sout);
         for (; sout[i]; i++);
      }
      return(sout);

   }
   else
   {
      *plen = 0;
      free(sout);
   }
   return(NULL);
}

static char *ATL_fgets_CWS(char *sout, int *plen, FILE *fpin)
/*
 * This routine returns a pointer to a single line of of file fpin.
 * It then compresses the whitespace in the line for ease of parsing:
 * (1) The first character in the line is non-whitespace
 * (2) The last character in the line is non-whitespace
 * (3) Any whitespace string of 1 or more ws chars is replaced with 1 ' '
 * (4) If the entire line is whitespace, get another until EOF or non-ws
 * If the size-len string sout is long enough to hold the file's line,
 * then sout will be the return value.  Otherwise sout will be freed and
 * a new string will be returned.
 * Upon EOF/error: sout is de-allocated, *len=0, & NULL is returned;
 * *len is the length of sout in input, and of the returned string on output.
 */
{
   int i, j;
   char *sp;
/*
 * Find the end of any preceding whitespace line; if the whole line is
 * whitespace, keep getting lines until we've got one with some non-ws chars
 */
   do
   {
      sout = ATL_fgets(sout, plen, fpin);
      if (!sout)
         return(NULL);
      for (i=0; isspace(sout[i]); i++);
   }
   while (sout[i] == '\0');
/*
 * Now, go through line, replacing all whitespace with single ' '
 */
   for (sp=sout+i,j=0; sp[j]; j++)
   {
      if (isspace(sp[j]))
      {
         sout[j] = ' ';
         while (isspace(sp[j])) sp++;
         sp--;
      }
      else
         sout[j] = sp[j];
   }
/*
 * Shave off any trailing ws (can only be one due to above)
 */
   if (isspace(sout[j-1]))
      sout[j-1] = '\0';
   else
      sout[j] = '\0';
   return(sout);
}

static char *ATL_tmpnam(void)
{
   static char tnam[L_tmpnam];
   static char FirstTime=1;
   if (FirstTime)
   {
      FirstTime = 0;
      assert(tmpnam(tnam));
   }
   return(tnam);
}

static FILE *atlsys(char *targ, char *cmnd, int verb, int IgnoreErr)
/*
 * Executes command cmnd, returns open ("r" mode) file stream to output of
 * command.  If IgnoreErr is 0, then return NULL on error.
 */
{
   char *tnam, *sp;
   int i;
   FILE *output=NULL;

   tnam = ATL_tmpnam();
   if (targ)
   {
      i = strlen(targ) + strlen(cmnd) + strlen(tnam) + 24;
      sp = malloc(i*sizeof(char));
      assert(sp);
      sprintf(sp, "ssh %s \"%s\" > %s 2>&1 \n", targ, cmnd, tnam);
   }
   else
   {
      i = strlen(cmnd) + strlen(tnam) + 16;
      sp = malloc(i*sizeof(char));
      assert(sp);
      sprintf(sp, "%s > %s 2>&1\n", cmnd, tnam);
   }
   i = system(sp);
   if (i && verb)
   {
      fprintf(stderr, "\nierr=%d in command='%s'!\n\n", i, cmnd);
      if (verb > 1)
      {
         fprintf(stderr, "OUTPUT:\n=======\n");
         sprintf(sp, "cat %s", tnam);
         system(sp);
      }
   }
   free(sp);
   if (!i || IgnoreErr)
      output = fopen(tnam, "r");
   return(output);
}

static char *atlsys_1L(char *targ, char *cmnd, int verb, int CWS)
/*
 * Executes system(cmnd), returns 1st line as allocated string.  Returns NULL
 * on error.
 */
{
   FILE *fp;
   char *ln=NULL;
   int len=0;

   fp = atlsys(targ, cmnd, verb, 0);
   if (fp)
   {
      if (CWS)
         ln = ATL_fgets_CWS(ln, &len, fp);
      else
         ln = ATL_fgets(ln, &len, fp);
      fclose(fp);
   }
   return(ln);
}
#endif
